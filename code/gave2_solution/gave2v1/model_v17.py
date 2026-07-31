from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ResidualDSBlock, _groups
from .model_v2 import GAVEV2, build_v2


class VeinGeometryRefiner(nn.Module):
    """Low-capacity full-resolution residual head for C-zone vein geometry."""

    def __init__(self, input_channels: int = 10) -> None:
        super().__init__()
        self.full = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(16), 16),
            nn.GELU(),
            ResidualDSBlock(16, 16),
        )
        self.half_stage = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(24), 24),
            nn.GELU(),
            ResidualDSBlock(24, 24),
        )
        self.quarter_stage = nn.Sequential(
            nn.Conv2d(24, 40, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(40), 40),
            nn.GELU(),
            ResidualDSBlock(40, 40),
        )
        self.eighth_stage = nn.Sequential(
            nn.Conv2d(40, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
            ResidualDSBlock(64, 64),
            ResidualDSBlock(64, 64),
        )
        self.up_quarter = nn.Sequential(
            ResidualDSBlock(64 + 40, 48),
            ResidualDSBlock(48, 40),
        )
        self.up_half = nn.Sequential(
            ResidualDSBlock(40 + 24, 32),
            ResidualDSBlock(32, 24),
        )
        self.up_full = nn.Sequential(
            ResidualDSBlock(24 + 16, 24),
            ResidualDSBlock(24, 16),
        )
        self.output = nn.Conv2d(16, 1, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small, non-zero initialization preserves the parent prediction while
        # allowing every refiner stage to receive gradients on the first step.
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        full = self.full(inputs)
        half = self.half_stage(full)
        quarter = self.quarter_stage(half)
        eighth = self.eighth_stage(quarter)
        feature = F.interpolate(
            eighth,
            size=quarter.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = self.up_quarter(torch.cat((feature, quarter), dim=1))
        feature = F.interpolate(
            feature,
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
        feature = self.up_full(torch.cat((feature, full), dim=1))
        return self.output(feature)


class GAVEV17Task3(nn.Module):
    """Frozen V8 Task2 plus a dedicated C-zone vein geometry head.

    The parent is immutable and remains the Task2 submission model. The new
    head changes only the Task3 vein source inside the SIVA C zone.
    """

    architecture_name = "GAVEV17-FrozenV8-CZoneVeinGeometry"

    def __init__(self, parent: GAVEV2) -> None:
        super().__init__()
        if parent.task != 2:
            raise ValueError("V17 requires a Task2 V8 parent")
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        self.refiner = VeinGeometryRefiner(input_channels=10)

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def parent_probabilities(self, rgb: Tensor, ffa: Tensor) -> Tensor:
        self.parent.eval()
        with torch.no_grad():
            outputs = self.parent(rgb, ffa)
            probabilities = self.parent.challenge_probabilities(outputs)
        return probabilities.detach()

    def forward(
        self,
        rgb: Tensor,
        ffa: Tensor,
        region: Tensor,
        *,
        refiner: nn.Module | None = None,
        apply_refiner: bool = True,
    ) -> dict[str, Tensor]:
        if ffa is None:
            raise ValueError("V17 requires registered FFA")
        if region.ndim != 4 or region.shape[1] != 1:
            raise ValueError("V17 region must have shape [B, 1, H, W]")
        parent_probabilities = self.parent_probabilities(rgb, ffa)
        parent_vein = parent_probabilities[:, 2:3].clamp(
            1e-4,
            1.0 - 1e-4,
        )
        parent_vein_logits = torch.logit(parent_vein)
        if apply_refiner:
            features = torch.cat(
                (rgb, ffa, parent_probabilities, region),
                dim=1,
            )
            residual_logits = (refiner or self.refiner)(features)
            refined_logits = parent_vein_logits + residual_logits
        else:
            residual_logits = torch.zeros_like(parent_vein_logits)
            refined_logits = parent_vein_logits
        refined_probability = torch.sigmoid(refined_logits)
        # C-zone metrics cannot depend on arbitrary behavior elsewhere.
        vein_probability = (
            region * refined_probability
            + (1.0 - region) * parent_vein
        )
        challenge_probabilities = torch.cat(
            (
                parent_probabilities[:, :2],
                vein_probability,
            ),
            dim=1,
        )
        return {
            "parent_probabilities": parent_probabilities,
            "residual_logits": residual_logits,
            "vein_logits": refined_logits,
            "vein_probability": vein_probability,
            "challenge_probabilities": challenge_probabilities,
        }

    def compact_state_dict(self) -> dict[str, Tensor]:
        return self.refiner.state_dict()


def load_v8_parent(
    checkpoint_path: str | Path,
) -> tuple[GAVEV2, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if int(config["task"]) != 2:
        raise ValueError("V17 parent checkpoint must be Task2")
    parent = build_v2(
        task=2,
        pretrained=False,
        num_refinements=int(config["num_refinements"]),
    )
    missing, unexpected = parent.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    if missing or unexpected:
        raise ValueError(
            f"V8 parent load failed: missing={missing}, unexpected={unexpected}"
        )
    report = {
        "checkpoint": str(path),
        "architecture": checkpoint.get("architecture"),
        "epoch": int(checkpoint["epoch"]),
        "loaded_tensors": len(checkpoint["model"]),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "height": int(config["height"]),
        "width": int(config["width"]),
        "num_refinements": int(config["num_refinements"]),
    }
    return parent, report


def load_compact_v17_state(
    model: GAVEV17Task3,
    state: Mapping[str, Tensor],
) -> dict[str, int]:
    missing, unexpected = model.refiner.load_state_dict(
        state,
        strict=True,
    )
    if missing or unexpected:
        raise ValueError(
            f"V17 refiner load failed: missing={missing}, unexpected={unexpected}"
        )
    return {
        "loaded_tensors": len(state),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }
