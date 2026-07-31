from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ConvNeXtFeatures, _groups
from .model_v2 import GAVEV2, warm_start_v2


class ResidualPhaseFusion(nn.Module):
    """Inject a full-capacity FFA feature without erasing the parent feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 8)
        self.phase_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        # The additional branch starts as a small, non-zero residual. This
        # preserves the warm-started parent while allowing gradients to reach
        # the phase encoder from the first optimizer step.
        nn.init.kaiming_normal_(
            self.phase_projection[0].weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.kaiming_normal_(
            self.gate[0].weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.gate[3].bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        self.strength_logit = nn.Parameter(torch.tensor(-2.1972246))

    def strength(self) -> Tensor:
        return torch.sigmoid(self.strength_logit)

    def forward(self, parent: Tensor, phase: Tensor) -> Tensor:
        phase = self.phase_projection(phase)
        gate = self.gate(torch.cat((parent, phase), dim=1))
        residual = self.output(phase * gate)
        return parent + self.strength() * residual


class GAVEV12(GAVEV2):
    """Task3 geometry model with a full pretrained FFA feature encoder.

    The historical lightweight FFA pyramid and its learned parent fusions are
    kept intact. A second full ConvNeXt-Tiny phase encoder reads registered
    [early, late, positive-difference] FFA and adds a small residual at every
    encoder scale. No biomarker regression head or validation-specific rule is
    used; Task3 values remain deterministic functions of dense A/V geometry.
    """

    architecture_name = "GAVEV12-DualScaleFullPhaseEncoder-RecurrentTopology"

    # Statistics were computed once from the 40 Fold0 training inputs after
    # the dataset's fixed robust FFA normalization. No validation labels or
    # images contributed.
    phase_mean = (0.38323483, 0.43301299, 0.10552736)
    phase_std = (0.26736039, 0.27849379, 0.15473673)

    def __init__(
        self,
        *,
        phase_pretrained: bool = True,
        num_refinements: int = 3,
    ) -> None:
        super().__init__(
            task=2,
            pretrained=False,
            num_refinements=num_refinements,
        )
        self.phase_encoder = ConvNeXtFeatures(pretrained=phase_pretrained)
        self.phase_fusions = nn.ModuleList(
            ResidualPhaseFusion(channels)
            for channels in self.encoder.channels
        )
        self.register_buffer(
            "_phase_mean",
            torch.tensor(self.phase_mean).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "_phase_std",
            torch.tensor(self.phase_std).view(1, 3, 1, 1),
            persistent=True,
        )

    def _encode_phase(self, ffa: Tensor) -> list[Tensor]:
        phase = (ffa - self._phase_mean) / self._phase_std
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.phase_encoder.features):
            phase = self._run_checkpointed(layer, phase)
            if index in (1, 3, 5, 7):
                outputs.append(phase)
        return outputs

    def _fuse_modalities(
        self,
        rgb_features: Sequence[Tensor],
        ffa: Tensor,
    ) -> list[Tensor]:
        parent_features = super()._fuse_modalities(rgb_features, ffa)
        phase_features = self._encode_phase(ffa)
        return [
            self._run_checkpointed(fusion, parent, phase)
            for fusion, parent, phase in zip(
                self.phase_fusions,
                parent_features,
                phase_features,
            )
        ]

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        encoder_parameters = [
            *self.encoder.parameters(),
            *self.phase_encoder.parameters(),
        ]
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in encoder_ids
        ]
        return encoder_parameters, new_parameters

    def phase_fusion_strengths(self) -> Tensor:
        return torch.stack(
            [fusion.strength() for fusion in self.phase_fusions]
        )


def build_v12(
    *,
    phase_pretrained: bool = True,
    num_refinements: int = 3,
) -> GAVEV12:
    return GAVEV12(
        phase_pretrained=phase_pretrained,
        num_refinements=num_refinements,
    )


def warm_start_v12(
    model: GAVEV12,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    report: dict[str, object] = {
        **warm_start_v2(model, checkpoint_path),
        "phase_encoder_source": "torchvision ImageNet ConvNeXt-Tiny",
    }
    return report
