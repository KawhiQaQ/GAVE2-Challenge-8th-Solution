from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import DetailStem
from .model_v2 import GAVEV2
from .model_v28 import NativeFinalAVRefiner


class GAVEV30(GAVEV2):
    """Task-specific, deterministic retinal physiology detail channels."""

    architecture_name = "GAVEV30-PhysiologyDetail-RecurrentTopology"
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        task: int,
        pretrained: bool = True,
        num_refinements: int = 3,
    ) -> None:
        super().__init__(
            task=task,
            pretrained=pretrained,
            num_refinements=num_refinements,
        )
        # Task1: RGB + red/green local optical density.
        # Task2: RGB + early/late/delta + two phase-balance channels.
        self.detail = DetailStem(5 if task == 1 else 8)
        self.native_final_refiner = (
            NativeFinalAVRefiner() if task == 1 else None
        )

    @staticmethod
    def _local_optical_density(
        channel: Tensor,
        kernel_size: int,
    ) -> Tensor:
        background = F.avg_pool2d(
            channel,
            kernel_size=kernel_size,
            stride=8,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        background = F.interpolate(
            background,
            size=channel.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.log(
            (background + 0.02) / (channel + 0.02)
        ).clamp(-2.0, 2.0)

    def _build_detail_input(
        self,
        rgb: Tensor,
        ffa: Tensor | None,
    ) -> Tensor:
        if self.task == 1:
            if ffa is not None:
                raise ValueError("V30 Task1 is strictly CFP-only")
            red = (
                rgb[:, 0:1] * self.rgb_std[0] + self.rgb_mean[0]
            ).clamp(0.0, 1.0)
            green = (
                rgb[:, 1:2] * self.rgb_std[1] + self.rgb_mean[1]
            ).clamp(0.0, 1.0)
            red_optical_density = self._local_optical_density(
                red,
                kernel_size=41,
            )
            green_optical_density = self._local_optical_density(
                green,
                kernel_size=21,
            )
            return torch.cat(
                (
                    rgb,
                    red_optical_density,
                    green_optical_density,
                ),
                dim=1,
            )

        if ffa is None:
            raise ValueError("V30 Task2 requires registered FFA")
        early, late, delta = ffa.split(1, dim=1)
        arterial_balance = (early - delta).clamp(-1.0, 1.0)
        venous_fraction = (
            delta / (late + 0.05)
        ).clamp(0.0, 2.0)
        return torch.cat(
            (
                rgb,
                ffa,
                arterial_balance,
                venous_fraction,
            ),
            dim=1,
        )

    def _finalize_native_av_logits(
        self,
        decoded_full: Tensor,
        av_logits: Tensor,
        structured_vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        if self.native_final_refiner is None:
            return av_logits
        return self.native_final_refiner(
            decoded_full,
            av_logits,
            structured_vessel_probability,
            centerline_probability,
        )


def build_v30(
    *,
    task: int,
    pretrained: bool = True,
    num_refinements: int = 3,
) -> GAVEV30:
    return GAVEV30(
        task=task,
        pretrained=pretrained,
        num_refinements=num_refinements,
    )


def warm_start_v30(
    model: GAVEV30,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    """Load V1 while zero-extending only the new detail input slices."""
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    source = checkpoint["model"]
    target = model.state_dict()
    source_detail_channels = 3 if model.task == 1 else 6
    target_detail_channels = 5 if model.task == 1 else 8
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
            and value.shape[1] == source_detail_channels
            and target[name].shape[1] == target_detail_channels
        ):
            initialized = target[name].clone()
            initialized[:, :source_detail_channels].copy_(value)
            initialized[:, source_detail_channels:].zero_()
            matched[name] = initialized
            expanded += 1
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "expanded_physiology_detail_tensors": expanded,
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }
