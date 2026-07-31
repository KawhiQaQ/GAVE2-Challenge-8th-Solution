from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNeXtFeatures(nn.Module):
    """ConvNeXt-Tiny feature extractor at strides 4, 8, 16 and 32."""

    channels = (96, 192, 384, 768)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.features = backbone.features
        self.gradient_checkpointing = False

    def forward(self, image: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.features):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                image = checkpoint(layer, image, use_reentrant=False)
            else:
                image = layer(image)
            if index in (1, 3, 5, 7):
                outputs.append(image)
        return outputs


class ResidualDSBlock(nn.Module):
    """Batch-size-one friendly residual depthwise-separable block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.project = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.depthwise1 = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.depthwise2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False
        )
        self.norm3 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm4 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feature: Tensor) -> Tensor:
        residual = self.project(feature)
        feature = self.depthwise1(feature)
        feature = F.gelu(self.norm1(feature))
        feature = self.pointwise1(feature)
        feature = F.gelu(self.norm2(feature))
        feature = self.dropout(feature)
        feature = self.depthwise2(feature)
        feature = F.gelu(self.norm3(feature))
        feature = self.pointwise2(feature)
        feature = self.norm4(feature)
        return F.gelu(feature + residual)


class FFAPyramid(nn.Module):
    """Lightweight functional pyramid for early/late/difference FFA."""

    channels = (48, 96, 192, 384)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        previous = in_channels
        for level, channels in enumerate(self.channels):
            if level == 0:
                down = nn.Conv2d(previous, channels, kernel_size=4, stride=4, bias=False)
            else:
                down = nn.Conv2d(previous, channels, kernel_size=2, stride=2, bias=False)
            stages.append(
                nn.Sequential(
                    down,
                    nn.GroupNorm(_groups(channels), channels),
                    nn.GELU(),
                    ResidualDSBlock(channels, channels),
                    ResidualDSBlock(channels, channels),
                )
            )
            previous = channels
        self.stages = nn.ModuleList(stages)

    def forward(self, image: Tensor) -> list[Tensor]:
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image)
        return outputs


class GatedCrossModalFusion(nn.Module):
    """Injects aligned FFA evidence only where RGB and FFA features agree."""

    def __init__(self, rgb_channels: int, ffa_channels: int) -> None:
        super().__init__()
        hidden = max(16, rgb_channels // 8)
        self.ffa_projection = nn.Sequential(
            nn.Conv2d(ffa_channels, rgb_channels, 1, bias=False),
            nn.GroupNorm(_groups(rgb_channels), rgb_channels),
            nn.GELU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1),
            nn.Sigmoid(),
        )
        self.refine = ResidualDSBlock(rgb_channels, rgb_channels)

    def forward(self, rgb: Tensor, ffa: Tensor) -> Tensor:
        ffa = self.ffa_projection(ffa)
        joint = torch.cat((rgb, ffa), dim=1)
        gate = self.spatial_gate(joint) * self.channel_gate(joint)
        return self.refine(rgb + gate * ffa)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            ResidualDSBlock(out_channels * 2, out_channels, dropout=0.05),
            ResidualDSBlock(out_channels, out_channels),
        )

    def forward(self, feature: Tensor, skip: Tensor) -> Tensor:
        feature = F.interpolate(
            self.input_projection(feature),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip = self.skip_projection(skip)
        return self.fuse(torch.cat((feature, skip), dim=1))


class DetailStem(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.full = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            ResidualDSBlock(24, 24),
        )
        self.half_branch = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            ResidualDSBlock(48, 48),
            ResidualDSBlock(48, 48),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return self.full(image), self.half_branch(image)


class GAVEV1(nn.Module):
    """Strong structured retinal A/V segmentation model.

    Semantic classes are [background, artery, vein, overlap]. Challenge
    probabilities are derived as [artery, all vessels, vein].
    """

    def __init__(self, task: int, pretrained: bool = True) -> None:
        super().__init__()
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        self.task = task
        self.encoder = ConvNeXtFeatures(pretrained=pretrained)
        if task == 2:
            self.ffa_encoder = FFAPyramid(in_channels=3)
            self.fusions = nn.ModuleList(
                GatedCrossModalFusion(rgb, ffa)
                for rgb, ffa in zip(self.encoder.channels, self.ffa_encoder.channels)
            )
            detail_channels = 6
        else:
            self.ffa_encoder = None
            self.fusions = None
            detail_channels = 3

        self.decode16 = DecoderBlock(768, 384, 256)
        self.decode8 = DecoderBlock(256, 192, 160)
        self.decode4 = DecoderBlock(160, 96, 96)
        self.detail = DetailStem(detail_channels)
        self.to_half = nn.Conv2d(96, 64, 1, bias=False)
        self.fuse_half = nn.Sequential(
            ResidualDSBlock(64 + 48, 72, dropout=0.05),
            ResidualDSBlock(72, 64),
        )
        self.fuse_full = nn.Sequential(
            ResidualDSBlock(64 + 24, 72, dropout=0.05),
            ResidualDSBlock(72, 64),
        )

        self.semantic_head = nn.Conv2d(64, 4, 1)
        self.vessel_head = nn.Conv2d(64, 1, 1)
        self.centerline_head = nn.Conv2d(64, 3, 1)
        self.aux_semantic_head = nn.Conv2d(96, 4, 1)
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        encoder_ids = {id(module) for module in self.encoder.modules()}
        for module in self.modules():
            if id(module) in encoder_ids:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Start auxiliary vessel probabilities conservatively near background.
        for head in (
            self.semantic_head,
            self.vessel_head,
            self.centerline_head,
            self.aux_semantic_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.vessel_head.bias, -2.0)
        nn.init.constant_(self.centerline_head.bias, -4.0)
        semantic_bias = self.semantic_head.bias.new_tensor((2.0, -1.5, -1.5, -3.0))
        self.semantic_head.bias.data.copy_(semantic_bias)
        self.aux_semantic_head.bias.data.copy_(semantic_bias)

    def _fuse_modalities(self, rgb_features: Sequence[Tensor], ffa: Tensor) -> list[Tensor]:
        if self.ffa_encoder is None or self.fusions is None:
            raise RuntimeError("FFA was supplied to a Task 1 model")
        ffa_features = self.ffa_encoder(ffa)
        return [
            fusion(rgb_feature, ffa_feature)
            for fusion, rgb_feature, ffa_feature in zip(
                self.fusions, rgb_features, ffa_features
            )
        ]

    def forward(self, rgb: Tensor, ffa: Tensor | None = None) -> dict[str, Tensor]:
        rgb_features = self.encoder(rgb)
        if self.task == 2:
            if ffa is None:
                raise ValueError("Task 2 requires a 3-channel FFA tensor")
            features = self._fuse_modalities(rgb_features, ffa)
            detail_input = torch.cat((rgb, ffa), dim=1)
        else:
            if ffa is not None:
                raise ValueError("Task 1 must not receive FFA")
            features = rgb_features
            detail_input = rgb

        stride4, stride8, stride16, stride32 = features
        decoded16 = self.decode16(stride32, stride16)
        decoded8 = self.decode8(decoded16, stride8)
        decoded4 = self.decode4(decoded8, stride4)
        detail_full, detail_half = self.detail(detail_input)
        decoded_half = F.interpolate(
            self.to_half(decoded4),
            size=detail_half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_half = self.fuse_half(torch.cat((decoded_half, detail_half), dim=1))
        decoded_full = F.interpolate(
            decoded_half,
            size=detail_full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_full = self.fuse_full(torch.cat((decoded_full, detail_full), dim=1))

        return {
            "semantic_logits": self.semantic_head(decoded_full),
            "vessel_logits": self.vessel_head(decoded_full),
            "centerline_logits": self.centerline_head(decoded_full),
            "aux_semantic_logits": self.aux_semantic_head(decoded4),
        }

    @staticmethod
    def challenge_probabilities(outputs: dict[str, Tensor]) -> Tensor:
        semantic = torch.softmax(outputs["semantic_logits"], dim=1)
        background, artery, vein, overlap = semantic.unbind(dim=1)
        semantic_vessel = 1.0 - background
        auxiliary_vessel = torch.sigmoid(outputs["vessel_logits"][:, 0])
        vessel = torch.maximum(semantic_vessel, auxiliary_vessel)
        artery = artery + overlap
        vein = vein + overlap
        return torch.stack((artery, vessel, vein), dim=1)

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        encoder_parameters = list(self.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        new_parameters = [
            parameter for parameter in self.parameters() if id(parameter) not in encoder_ids
        ]
        return encoder_parameters, new_parameters


def build_v1(task: int, pretrained: bool = True) -> GAVEV1:
    return GAVEV1(task=task, pretrained=pretrained)


def warm_start_from_checkpoint(
    model: GAVEV1,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    """Load every compatible tensor and expand RGB detail kernels for Task 2."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    source = checkpoint["model"]
    target = model.state_dict()
    matched: dict[str, Tensor] = {}
    expanded = 0
    for name, value in source.items():
        if name not in target:
            continue
        if target[name].shape == value.shape:
            matched[name] = value
            continue
        if (
            value.ndim == 4
            and target[name].ndim == 4
            and value.shape[0] == target[name].shape[0]
            and value.shape[2:] == target[name].shape[2:]
            and value.shape[1] == 3
            and target[name].shape[1] == 6
        ):
            initialized = target[name].clone()
            initialized[:, :3].copy_(value)
            initialized[:, 3:].zero_()
            matched[name] = initialized
            expanded += 1
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "expanded_rgb_to_multimodal": expanded,
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }
