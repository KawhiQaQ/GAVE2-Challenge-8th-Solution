from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import (
    ConvNeXtFeatures,
    DecoderBlock,
    DetailStem,
    FFAPyramid,
    GatedCrossModalFusion,
    ResidualDSBlock,
    _groups,
)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)

    def forward(self, feature: Tensor) -> Tensor:
        residual = feature
        feature = F.gelu(self.norm1(self.depthwise(feature)))
        feature = self.norm2(self.pointwise(feature))
        return F.gelu(feature + residual)


class RecurrentTopologyRefiner(nn.Module):
    """Shared multi-scale A/V refiner conditioned on a fixed vessel map."""

    def __init__(self, context_channels: int = 12) -> None:
        super().__init__()
        input_channels = context_channels + 3 + 2
        self.full = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            ResidualDSBlock(16, 16),
        )
        self.down_half = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            ResidualDSBlock(24, 24),
        )
        self.down_quarter = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            DilatedResidualBlock(32, dilation=2),
            DilatedResidualBlock(32, dilation=4),
        )
        self.up_half = nn.Sequential(
            ResidualDSBlock(32 + 24, 32),
            ResidualDSBlock(32, 24),
        )
        self.up_full = nn.Sequential(
            ResidualDSBlock(24 + 16, 24),
            ResidualDSBlock(24, 16),
        )
        self.output = nn.Conv2d(16, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        context: Tensor,
        av_probability: Tensor,
        vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        state = torch.cat(
            (
                context,
                av_probability,
                vessel_probability,
                centerline_probability,
            ),
            dim=1,
        )
        full = self.full(state)
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        feature = F.interpolate(
            quarter,
            size=half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = self.up_half(torch.cat((feature, half), dim=1))
        feature = F.interpolate(
            feature,
            size=full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output(self.up_full(torch.cat((feature, full), dim=1)))


class RetinalVesselNet(nn.Module):
    """Native-resolution recurrent structured A/V segmentation network."""

    architecture_name = "VascFusion-RetinalVesselCore"

    def __init__(
        self,
        task: int,
        pretrained: bool = True,
        num_refinements: int = 3,
    ) -> None:
        super().__init__()
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        self.task = task
        self.num_refinements = num_refinements
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

        # Stage-one-compatible heads make warm-starting lossless at iteration zero.
        self.semantic_head = nn.Conv2d(64, 4, 1)
        self.vessel_head = nn.Conv2d(64, 1, 1)
        self.centerline_head = nn.Conv2d(64, 3, 1)
        self.aux_semantic_head = nn.Conv2d(96, 4, 1)

        self.av_residual_head = nn.Conv2d(64, 2, 1)
        self.refinement_context = nn.Sequential(
            nn.Conv2d(64, 12, 1, bias=False),
            nn.GroupNorm(4, 12),
            nn.GELU(),
            ResidualDSBlock(12, 12),
        )
        self.topology_refiner = RecurrentTopologyRefiner(context_channels=12)
        self._initialize_new_layers()

    def _run_checkpointed(self, module: nn.Module, *inputs: Tensor):
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _encode_rgb(self, image: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.encoder.features):
            image = self._run_checkpointed(layer, image)
            if index in (1, 3, 5, 7):
                outputs.append(image)
        return outputs

    def _initialize_new_layers(self) -> None:
        encoder_ids = {id(module) for module in self.encoder.modules()}
        for module in self.modules():
            if id(module) in encoder_ids:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for head in (
            self.semantic_head,
            self.vessel_head,
            self.centerline_head,
            self.aux_semantic_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.vessel_head.bias, -2.0)
        nn.init.constant_(self.centerline_head.bias, -4.0)
        semantic_bias = self.semantic_head.bias.new_tensor(
            (2.0, -1.5, -1.5, -3.0)
        )
        self.semantic_head.bias.data.copy_(semantic_bias)
        self.aux_semantic_head.bias.data.copy_(semantic_bias)

        # The recursive head initially preserves the semantic marginals.
        nn.init.zeros_(self.av_residual_head.weight)
        nn.init.zeros_(self.av_residual_head.bias)
        nn.init.zeros_(self.topology_refiner.output.weight)
        nn.init.zeros_(self.topology_refiner.output.bias)

    def _fuse_modalities(
        self,
        rgb_features: Sequence[Tensor],
        ffa: Tensor,
    ) -> list[Tensor]:
        if self.ffa_encoder is None or self.fusions is None:
            raise RuntimeError("FFA was supplied to a Task 1 model")
        ffa_features: list[Tensor] = []
        feature = ffa
        for stage in self.ffa_encoder.stages:
            feature = self._run_checkpointed(stage, feature)
            ffa_features.append(feature)
        return [
            self._run_checkpointed(
                fusion,
                rgb_feature,
                ffa_feature,
            )
            for fusion, rgb_feature, ffa_feature in zip(
                self.fusions,
                rgb_features,
                ffa_features,
            )
        ]

    def _build_detail_input(
        self,
        rgb: Tensor,
        ffa: Tensor | None,
    ) -> Tensor:
        if self.task == 2:
            if ffa is None:
                raise ValueError("Task 2 requires FFA detail input")
            return torch.cat((rgb, ffa), dim=1)
        return rgb

    def _condition_av_logits(
        self,
        av_logits: Tensor,
        structured_vessel_probability: Tensor,
    ) -> Tensor:
        """Optional architecture hook for vessel-conditioned A/V outputs."""
        return av_logits

    def _condition_av_logits_with_centerline(
        self,
        av_logits: Tensor,
        structured_vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        """Backward-compatible hook for class-specific centerline evidence."""
        del centerline_probability
        return self._condition_av_logits(
            av_logits,
            structured_vessel_probability,
        )

    def _finalize_native_av_logits(
        self,
        decoded_full: Tensor,
        av_logits: Tensor,
        structured_vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        """Optional final native-resolution correction hook."""
        del decoded_full, structured_vessel_probability, centerline_probability
        return av_logits

    def _extra_outputs(
        self,
        decoded_full: Tensor,
        av_logits: list[Tensor],
        centerline_logits: Tensor,
        structured_vessel_probability: Tensor,
    ) -> dict[str, Tensor]:
        """Optional version-specific outputs derived during the main forward."""
        del (
            decoded_full,
            av_logits,
            centerline_logits,
            structured_vessel_probability,
        )
        return {}

    def _augment_refinement_context(
        self,
        context: Tensor,
        rgb: Tensor,
        ffa: Tensor | None,
    ) -> Tensor:
        """Optional fixed image/modality evidence for every recurrent step."""
        del rgb, ffa
        return context

    def _refine_av_logits(
        self,
        context: Tensor,
        coarse_av_logits: Tensor,
        structured_vessel_probability: Tensor,
        vessel_probability_half: Tensor,
        centerline_probability: Tensor,
        centerline_probability_half: Tensor,
    ) -> list[Tensor]:
        """Run the architecture's shared A/V recurrence.

        This hook preserves the core computation while allowing
        later models to replace the recurrent core without copying the complete
        encoder/decoder forward.
        """
        latent_av_logits = coarse_av_logits
        current_logits = self._condition_av_logits_with_centerline(
            latent_av_logits,
            structured_vessel_probability,
            centerline_probability,
        )
        av_logits: list[Tensor] = [current_logits]
        refinement_size = context.shape[-2:]
        for _ in range(self.num_refinements):
            current_probability_half = F.interpolate(
                torch.sigmoid(current_logits),
                size=refinement_size,
                mode="bilinear",
                align_corners=False,
            )
            correction_half = self._run_checkpointed(
                self.topology_refiner,
                context,
                current_probability_half,
                vessel_probability_half,
                centerline_probability_half,
            )
            correction = F.interpolate(
                correction_half,
                size=current_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            latent_av_logits = latent_av_logits + correction
            current_logits = self._condition_av_logits_with_centerline(
                latent_av_logits,
                structured_vessel_probability,
                centerline_probability,
            )
            av_logits.append(current_logits)
        return av_logits

    def forward(self, rgb: Tensor, ffa: Tensor | None = None) -> dict[str, object]:
        rgb_features = self._encode_rgb(rgb)
        if self.task == 2:
            if ffa is None:
                raise ValueError("Task 2 requires a 3-channel FFA tensor")
            features = self._fuse_modalities(rgb_features, ffa)
        else:
            if ffa is not None:
                raise ValueError("Task 1 must not receive FFA")
            features = rgb_features
        detail_input = self._build_detail_input(rgb, ffa)

        stride4, stride8, stride16, stride32 = features
        decoded16 = self._run_checkpointed(
            self.decode16,
            stride32,
            stride16,
        )
        decoded8 = self._run_checkpointed(
            self.decode8,
            decoded16,
            stride8,
        )
        decoded4 = self._run_checkpointed(
            self.decode4,
            decoded8,
            stride4,
        )
        detail_full, detail_half = self._run_checkpointed(
            self.detail,
            detail_input,
        )
        decoded_half = F.interpolate(
            self.to_half(decoded4),
            size=detail_half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_half = self._run_checkpointed(
            self.fuse_half,
            torch.cat((decoded_half, detail_half), dim=1),
        )
        decoded_full = F.interpolate(
            decoded_half,
            size=detail_full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_full = self._run_checkpointed(
            self.fuse_full,
            torch.cat((decoded_full, detail_full), dim=1),
        )

        semantic_logits = self.semantic_head(decoded_full)
        semantic_probability = torch.softmax(semantic_logits, dim=1)
        _, artery, vein, overlap = semantic_probability.unbind(dim=1)
        semantic_av_probability = torch.stack(
            (artery + overlap, vein + overlap),
            dim=1,
        ).clamp(1e-4, 1.0 - 1e-4)
        coarse_av_logits = torch.logit(semantic_av_probability)
        coarse_av_logits = coarse_av_logits + self.av_residual_head(decoded_full)

        vessel_logits = self.vessel_head(decoded_full)
        vessel_probability = torch.sigmoid(vessel_logits)
        semantic_vessel_probability = 1.0 - semantic_probability[:, :1]
        structured_vessel_probability = torch.maximum(
            semantic_vessel_probability,
            vessel_probability,
        )
        centerline_logits = self.centerline_head(decoded_full)
        centerline_probability = torch.sigmoid(
            centerline_logits[:, (0, 2)]
        )
        # Refine on the native decoder's half-resolution feature map, then
        # write corrections back to full resolution. This retains native
        # full-resolution logits while keeping recurrent training tractable.
        context = self._run_checkpointed(
            self.refinement_context,
            decoded_half,
        )
        context = self._augment_refinement_context(context, rgb, ffa)
        refinement_size = context.shape[-2:]
        vessel_probability_half = F.interpolate(
            vessel_probability,
            size=refinement_size,
            mode="bilinear",
            align_corners=False,
        )
        centerline_probability_half = F.interpolate(
            centerline_probability,
            size=refinement_size,
            mode="bilinear",
            align_corners=False,
        )

        av_logits = self._refine_av_logits(
            context,
            coarse_av_logits,
            structured_vessel_probability,
            vessel_probability_half,
            centerline_probability,
            centerline_probability_half,
        )

        av_logits[-1] = self._finalize_native_av_logits(
            decoded_full,
            av_logits[-1],
            structured_vessel_probability,
            centerline_probability,
        )

        outputs: dict[str, object] = {
            "semantic_logits": semantic_logits,
            "vessel_logits": vessel_logits,
            "centerline_logits": centerline_logits,
            "aux_semantic_logits": self.aux_semantic_head(decoded4),
            "av_logits": av_logits,
        }
        outputs.update(
            self._extra_outputs(
                decoded_full,
                av_logits,
                centerline_logits,
                structured_vessel_probability,
            )
        )
        return outputs

    @staticmethod
    def challenge_probabilities(outputs: dict[str, object]) -> Tensor:
        av_logits = outputs["av_logits"]
        if not isinstance(av_logits, list):
            raise TypeError("outputs['av_logits'] must be a recurrent list")
        artery, vein = torch.sigmoid(av_logits[-1]).unbind(dim=1)
        semantic_logits = outputs["semantic_logits"]
        vessel_logits = outputs["vessel_logits"]
        if not isinstance(semantic_logits, Tensor) or not isinstance(
            vessel_logits,
            Tensor,
        ):
            raise TypeError("semantic and vessel logits must be tensors")
        semantic_vessel = 1.0 - torch.softmax(semantic_logits, dim=1)[:, 0]
        vessel = torch.maximum(
            semantic_vessel,
            torch.sigmoid(vessel_logits[:, 0]),
        )
        return torch.stack((artery, vessel, vein), dim=1)

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        encoder_parameters = list(self.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in encoder_ids
        ]
        return encoder_parameters, new_parameters


def build_retinal_vessel_net(
    task: int,
    pretrained: bool = True,
    num_refinements: int = 3,
) -> RetinalVesselNet:
    return RetinalVesselNet(
        task=task,
        pretrained=pretrained,
        num_refinements=num_refinements,
    )


def warm_start_retinal_vessel_net(
    model: RetinalVesselNet,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
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
