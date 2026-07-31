from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import DetailStem, ResidualDSBlock, _groups
from .model_v2 import DilatedResidualBlock, GAVEV2, warm_start_v2
from .model_v28 import NativeFinalAVRefiner
from .model_v30 import warm_start_v30


class R2SharedProbabilityUNet(nn.Module):
    """Deep shared U-Net that recursively classifies a fixed vessel tree."""

    def __init__(self) -> None:
        super().__init__()
        # Current artery, current vein and fixed generic vessel.
        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
            ResidualDSBlock(32, 32),
            ResidualDSBlock(32, 32),
        )
        self.down_half = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(48), 48),
            nn.GELU(),
            ResidualDSBlock(48, 48),
            ResidualDSBlock(48, 48),
        )
        self.down_quarter = nn.Sequential(
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
            DilatedResidualBlock(64, dilation=2),
            DilatedResidualBlock(64, dilation=4),
        )
        self.down_eighth = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            DilatedResidualBlock(96, dilation=2),
            DilatedResidualBlock(96, dilation=4),
        )
        self.up_quarter = nn.Sequential(
            ResidualDSBlock(96 + 64, 96),
            ResidualDSBlock(96, 64),
        )
        self.up_half = nn.Sequential(
            ResidualDSBlock(64 + 48, 64),
            ResidualDSBlock(64, 48),
        )
        self.up_full = nn.Sequential(
            ResidualDSBlock(48 + 32, 48),
            ResidualDSBlock(48, 32),
        )
        self.output = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        av_probability: Tensor,
        vessel_probability: Tensor,
    ) -> Tensor:
        full = self.full(torch.cat((av_probability, vessel_probability), dim=1))
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        eighth = self.down_eighth(quarter)

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
        return self.output(self.up_full(torch.cat((feature, full), dim=1)))


class GAVEV54(GAVEV2):
    """Task-specific best geometry plus a deep R2 probability-only U-Net."""

    architecture_name = "GAVEV54-DeepR2SharedProbabilityUNet"
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        task: int,
        pretrained: bool = True,
        r2_steps: int = 5,
    ) -> None:
        if r2_steps < 1:
            raise ValueError("r2_steps must be positive")
        super().__init__(
            task=task,
            pretrained=pretrained,
            num_refinements=r2_steps,
        )
        self.r2_steps = r2_steps
        if task == 1:
            self.detail = DetailStem(5)
            self.native_final_refiner: NativeFinalAVRefiner | None = (
                NativeFinalAVRefiner()
            )
        else:
            self.native_final_refiner = None

        # Replace, rather than stack on, the old context refiner.
        self.refinement_context = nn.Identity()
        self.topology_refiner = nn.Identity()
        self.r2_specialist = R2SharedProbabilityUNet()
        self._initialize_v54_layers()

    def _initialize_v54_layers(self) -> None:
        modules: list[nn.Module] = [self.r2_specialist]
        if self.task == 1:
            modules.extend((self.detail, self.native_final_refiner))
        for root in modules:
            if root is None:
                continue
            for module in root.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.zeros_(self.r2_specialist.output.weight)
        nn.init.zeros_(self.r2_specialist.output.bias)
        if self.native_final_refiner is not None:
            nn.init.zeros_(self.native_final_refiner.output.weight)
            nn.init.zeros_(self.native_final_refiner.output.bias)

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
        if self.task == 2:
            return super()._build_detail_input(rgb, ffa)
        if ffa is not None:
            raise ValueError("V54 Task1 is strictly CFP-only")
        red = (
            rgb[:, 0:1] * self.rgb_std[0] + self.rgb_mean[0]
        ).clamp(0.0, 1.0)
        green = (
            rgb[:, 1:2] * self.rgb_std[1] + self.rgb_mean[1]
        ).clamp(0.0, 1.0)
        return torch.cat(
            (
                rgb,
                self._local_optical_density(red, kernel_size=41),
                self._local_optical_density(green, kernel_size=21),
            ),
            dim=1,
        )

    def _refine_av_logits(
        self,
        context: Tensor,
        coarse_av_logits: Tensor,
        structured_vessel_probability: Tensor,
        vessel_probability_half: Tensor,
        centerline_probability: Tensor,
        centerline_probability_half: Tensor,
    ) -> list[Tensor]:
        del context, vessel_probability_half, centerline_probability_half
        refinement_size = (
            coarse_av_logits.shape[-2] // 2,
            coarse_av_logits.shape[-1] // 2,
        )
        fixed_vessel = F.interpolate(
            structured_vessel_probability,
            size=refinement_size,
            mode="bilinear",
            align_corners=False,
        )
        latent_logits = coarse_av_logits
        current_logits = self._condition_av_logits_with_centerline(
            latent_logits,
            structured_vessel_probability,
            centerline_probability,
        )
        av_logits: list[Tensor] = [current_logits]
        for _ in range(self.r2_steps):
            current_probability = F.interpolate(
                torch.sigmoid(current_logits),
                size=refinement_size,
                mode="bilinear",
                align_corners=False,
            )
            correction_half = self._run_checkpointed(
                self.r2_specialist,
                current_probability,
                fixed_vessel,
            )
            correction = F.interpolate(
                correction_half,
                size=current_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            latent_logits = latent_logits + correction
            current_logits = self._condition_av_logits_with_centerline(
                latent_logits,
                structured_vessel_probability,
                centerline_probability,
            )
            av_logits.append(current_logits)
        return av_logits

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


def build_v54(
    *,
    task: int,
    pretrained: bool = True,
    r2_steps: int = 5,
) -> GAVEV54:
    return GAVEV54(
        task=task,
        pretrained=pretrained,
        r2_steps=r2_steps,
    )


def warm_start_v54(
    model: GAVEV54,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    if model.task == 1:
        return warm_start_v30(model, checkpoint_path)
    return warm_start_v2(model, checkpoint_path)

