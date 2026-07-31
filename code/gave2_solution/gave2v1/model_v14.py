from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
from typing import Any

import timm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ResidualDSBlock, _groups
from .model_v2 import GAVEV2, warm_start_v2


RETFOUND_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
RETFOUND_INPUT_SIZE = (364, 546)
RETFOUND_LAYER_INDICES = (23,)
RETFOUND_EXPECTED_SHA256 = (
    "a3feeaf69b44a0baa0934162d06b11ca"
    "517d1ca71a8459e04ef72c287d4e7124"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_retfound_prefix(name: str) -> str:
    for prefix in ("module.", "teacher.", "backbone."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("mlp.w12.", "mlp.fc1.").replace(
        "mlp.w3.",
        "mlp.fc2.",
    )


def _interpolate_position_embedding(
    position_embedding: Tensor,
    target_grid: tuple[int, int],
) -> Tensor:
    if position_embedding.ndim != 3 or position_embedding.shape[0] != 1:
        raise ValueError(
            "RETFound position embedding must have shape [1, tokens, channels]"
        )
    patch_tokens = int(position_embedding.shape[1]) - 1
    source_side = math.isqrt(patch_tokens)
    if source_side * source_side != patch_tokens:
        raise ValueError(
            "RETFound position embedding patch tokens must form a square grid"
        )
    prefix = position_embedding[:, :1]
    patches = position_embedding[:, 1:].reshape(
        1,
        source_side,
        source_side,
        position_embedding.shape[-1],
    )
    patches = patches.permute(0, 3, 1, 2)
    patches = F.interpolate(
        patches,
        size=target_grid,
        mode="bicubic",
        align_corners=False,
    )
    patches = patches.permute(0, 2, 3, 1).reshape(
        1,
        target_grid[0] * target_grid[1],
        position_embedding.shape[-1],
    )
    return torch.cat((prefix, patches), dim=1)


class FrozenRETFoundDinoV2(nn.Module):
    """Official retinal DINOv2 teacher used as an immutable feature source."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        input_size: tuple[int, int] = RETFOUND_INPUT_SIZE,
        layer_indices: Sequence[int] = RETFOUND_LAYER_INDICES,
        expected_sha256: str = RETFOUND_EXPECTED_SHA256,
    ) -> None:
        super().__init__()
        if any(value % 14 for value in input_size):
            raise ValueError("RETFound input dimensions must be divisible by 14")
        if tuple(sorted(layer_indices)) != tuple(layer_indices):
            raise ValueError("RETFound layer indices must be sorted")

        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RETFound checkpoint not found: {path}")
        actual_sha256 = sha256_file(path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                "RETFound checkpoint SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        self.input_size = tuple(int(value) for value in input_size)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = actual_sha256
        self.backbone = timm.create_model(
            RETFOUND_MODEL_NAME,
            pretrained=False,
            img_size=self.input_size,
            num_classes=0,
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping) or "teacher" not in checkpoint:
            raise ValueError("RETFound checkpoint must contain a teacher state")
        teacher = checkpoint["teacher"]
        if not isinstance(teacher, Mapping):
            raise ValueError("RETFound teacher state must be a mapping")
        source = {
            _strip_retfound_prefix(str(name)): value
            for name, value in teacher.items()
            if isinstance(value, Tensor)
        }
        if "pos_embed" not in source:
            raise ValueError("RETFound teacher is missing pos_embed")
        source["pos_embed"] = _interpolate_position_embedding(
            source["pos_embed"],
            tuple(self.backbone.patch_embed.grid_size),
        )

        target = self.backbone.state_dict()
        matched = {
            name: value
            for name, value in source.items()
            if name in target and target[name].shape == value.shape
        }
        covered_parameters = sum(value.numel() for value in matched.values())
        target_parameters = sum(value.numel() for value in target.values())
        coverage = covered_parameters / max(1, target_parameters)
        if coverage < 0.999:
            raise ValueError(
                f"RETFound backbone coverage too low: {coverage:.6%}"
            )
        missing, unexpected = self.backbone.load_state_dict(
            matched,
            strict=False,
        )
        if missing or unexpected:
            raise ValueError(
                "RETFound backbone did not load cleanly: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        self.load_report: dict[str, Any] = {
            "checkpoint": str(path),
            "sha256": actual_sha256,
            "model_name": RETFOUND_MODEL_NAME,
            "input_size": list(self.input_size),
            "patch_grid": list(self.backbone.patch_embed.grid_size),
            "layer_indices": list(self.layer_indices),
            "matched_tensors": len(matched),
            "target_tensors": len(target),
            "parameter_coverage": coverage,
            "source_position_tokens": int(
                teacher["backbone.pos_embed"].shape[1]
            ),
            "target_position_tokens": int(
                source["pos_embed"].shape[1]
            ),
        }
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def __deepcopy__(self, memo: dict[int, object]):
        # EMA must share this immutable 1.2 GB encoder rather than duplicating it.
        memo[id(self)] = self
        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        image = F.interpolate(
            image,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                image,
                n=self.layer_indices,
                reshape=True,
                return_prefix_tokens=False,
                norm=True,
            )
        return tuple(features)


class RetinalBottleneckAdapter(nn.Module):
    """Zero-initialized retinal context residual at the stride-32 bottleneck."""

    def __init__(
        self,
        rgb_channels: int,
        context_channels: int = 1024,
    ) -> None:
        super().__init__()
        hidden = 192
        self.context_projection = nn.Sequential(
            nn.Conv2d(context_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                rgb_channels + hidden,
                rgb_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(_groups(rgb_channels), rgb_channels),
            nn.GELU(),
            ResidualDSBlock(rgb_channels, rgb_channels),
        )
        self.output = nn.Conv2d(rgb_channels, rgb_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, rgb: Tensor, retinal_context: Tensor) -> Tensor:
        retinal_context = F.interpolate(
            retinal_context,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = self.context_projection(retinal_context)
        residual = self.output(
            self.fusion(torch.cat((rgb, context), dim=1))
        )
        return rgb + residual


class GAVEV14(GAVEV2):
    """V8 high-resolution model plus frozen retinal-domain DINOv2 context."""

    architecture_name = (
        "GAVEV14-ConvNeXtTiny-RETFoundDINOv2-RecurrentTopology"
    )

    def __init__(
        self,
        task: int,
        retfound_checkpoint: str | Path,
        pretrained: bool = True,
        num_refinements: int = 3,
        retfound_input_size: tuple[int, int] = RETFOUND_INPUT_SIZE,
    ) -> None:
        super().__init__(
            task=task,
            pretrained=pretrained,
            num_refinements=num_refinements,
        )
        self.retfound_encoder = FrozenRETFoundDinoV2(
            retfound_checkpoint,
            input_size=retfound_input_size,
        )
        self.retfound_bottleneck_adapter = RetinalBottleneckAdapter(
            self.encoder.channels[-1]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.retfound_encoder.eval()
        return self

    def _encode_rgb(self, image: Tensor) -> list[Tensor]:
        rgb_features = super()._encode_rgb(image)
        retinal_features = self.retfound_encoder(image)
        if len(retinal_features) != 1:
            raise RuntimeError("V14 expects one RETFound bottleneck feature")
        rgb_features[-1] = self._run_checkpointed(
            self.retfound_bottleneck_adapter,
            rgb_features[-1],
            retinal_features[0],
        )
        return rgb_features

    def parameter_groups(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        adapter_parameters = [
            parameter
            for parameter in self.retfound_bottleneck_adapter.parameters()
            if parameter.requires_grad
        ]
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        parent_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_ids
        ]
        # engine_v2 applies encoder_lr_ratio to the first group. For V14 this
        # intentionally gives the proven V8 parent a 10x lower learning rate
        # than the new zero-initialized bottleneck adapter.
        return parent_parameters, adapter_parameters

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        compact = OrderedDict(
            (name, value)
            for name, value in state.items()
            if not name.startswith("retfound_encoder.")
        )
        if hasattr(state, "_metadata"):
            compact._metadata = state._metadata
        return compact


def build_v14(
    task: int,
    retfound_checkpoint: str | Path,
    pretrained: bool = True,
    num_refinements: int = 3,
    retfound_input_size: tuple[int, int] = RETFOUND_INPUT_SIZE,
) -> GAVEV14:
    return GAVEV14(
        task=task,
        retfound_checkpoint=retfound_checkpoint,
        pretrained=pretrained,
        num_refinements=num_refinements,
        retfound_input_size=retfound_input_size,
    )


def warm_start_v14(
    model: GAVEV14,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    return warm_start_v2(model, checkpoint_path)


def load_compact_v14_state(
    model: GAVEV14,
    state: Mapping[str, Tensor],
) -> dict[str, int]:
    missing, unexpected = model.load_state_dict(state, strict=False)
    illegal_missing = [
        name
        for name in missing
        if not name.startswith("retfound_encoder.")
    ]
    if illegal_missing or unexpected:
        raise ValueError(
            "Invalid compact V14 checkpoint: "
            f"missing={illegal_missing[:10]}, unexpected={unexpected[:10]}"
        )
    return {
        "loaded_tensors": len(state),
        "retfound_tensors_restored_from_official_weight": len(missing),
        "unexpected_tensors": len(unexpected),
    }
