from __future__ import annotations

from torch import Tensor, nn
import torch

from .model import ResidualDSBlock
from .model_v2 import GAVEV2


class NativeFinalAVRefiner(nn.Module):
    """Lightweight native-resolution correction after recurrent refinement."""

    def __init__(self, decoder_channels: int = 64) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(decoder_channels, 8, 1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
        )
        # context + current A/V + generic vessel + class centerlines
        self.fuse = nn.Sequential(
            nn.Conv2d(8 + 2 + 1 + 2, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            ResidualDSBlock(16, 16),
        )
        self.output = nn.Conv2d(16, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        decoded_full: Tensor,
        av_logits: Tensor,
        vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        evidence = torch.cat(
            (
                self.context(decoded_full),
                av_logits.sigmoid().detach(),
                vessel_probability.detach(),
                centerline_probability.detach(),
            ),
            dim=1,
        )
        return av_logits + self.output(self.fuse(evidence))


class GAVEV28(GAVEV2):
    """V8 plus one zero-initialized final native-resolution A/V refiner."""

    architecture_name = "GAVEV28-NativeFinalAVRefiner"

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
        self.native_final_refiner = NativeFinalAVRefiner()

    def _finalize_native_av_logits(
        self,
        decoded_full: Tensor,
        av_logits: Tensor,
        structured_vessel_probability: Tensor,
        centerline_probability: Tensor,
    ) -> Tensor:
        return self.native_final_refiner(
            decoded_full,
            av_logits,
            structured_vessel_probability,
            centerline_probability,
        )


def build_v28(
    *,
    task: int,
    pretrained: bool = True,
    num_refinements: int = 3,
) -> GAVEV28:
    return GAVEV28(
        task=task,
        pretrained=pretrained,
        num_refinements=num_refinements,
    )
