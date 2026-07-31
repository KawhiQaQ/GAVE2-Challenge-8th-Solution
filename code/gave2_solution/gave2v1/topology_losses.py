from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .losses import (
    binary_focal_loss,
    focal_cross_entropy,
    focal_tversky_loss,
    semantic_probabilities,
    soft_cldice_loss,
    soft_dice_loss,
)


def conditional_av_loss(
    logits: Tensor,
    target: Tensor,
    classes: Tensor,
) -> Tensor:
    pure_vessel = (classes == 1) | (classes == 2)
    if not bool(pure_vessel.any()):
        return logits.sum() * 0.0
    expanded_mask = pure_vessel[:, None].expand_as(logits)
    return F.binary_cross_entropy_with_logits(
        logits[expanded_mask],
        target.to(logits.dtype)[expanded_mask],
    )


def centerline_coverage_loss(
    probability: Tensor,
    centerline: Tensor,
    roi: Tensor,
) -> Tensor:
    valid_centerline = centerline.to(probability.dtype) * roi
    numerator = (
        -torch.log(probability.clamp_min(1e-5)) * valid_centerline
    ).sum(dim=(0, 2, 3))
    denominator = valid_centerline.sum(dim=(0, 2, 3)).clamp_min(1.0)
    return (numerator / denominator).mean()


def hard_centerline_gap_loss(
    probability: Tensor,
    centerline: Tensor,
    roi: Tensor,
    *,
    hardest_fraction: float = 0.20,
    margin: float = 0.65,
) -> Tensor:
    """Lift the weakest GT centerline responses above the path threshold.

    Mean centerline coverage can stay low even when a few sub-threshold pixels
    disconnect a long branch. Focusing on the weakest fraction is a
    differentiable way to repair those bottlenecks without thickening every
    vessel pixel.
    """
    if not 0.0 < hardest_fraction <= 1.0:
        raise ValueError("hardest_fraction must be in (0, 1]")
    valid = centerline.bool() & (roi > 0.5)
    channel_losses: list[Tensor] = []
    for batch_index in range(probability.shape[0]):
        for channel_index in range(probability.shape[1]):
            values = probability[batch_index, channel_index][
                valid[batch_index, channel_index]
            ]
            if values.numel() == 0:
                continue
            violations = F.relu(margin - values)
            count = max(
                1,
                int(round(hardest_fraction * int(violations.numel()))),
            )
            channel_losses.append(
                torch.topk(violations, k=count, sorted=False).values.mean()
            )
    if not channel_losses:
        return probability.sum() * 0.0
    return torch.stack(channel_losses).mean()


@dataclass(frozen=True)
class TopologyLossWeights:
    recurrent_av: float = 1.0
    vessel: float = 0.9
    topology: float = 0.70
    # A non-zero value enables the weakest-centerline gap objective.
    # reserved for a separately validated later version.
    hard_gap: float = 0.0
    centerline: float = 0.25
    semantic: float = 0.30
    deep_supervision: float = 0.15
    hierarchy: float = 0.10


class TopologyAwareLoss(nn.Module):
    def __init__(
        self,
        weights: TopologyLossWeights = TopologyLossWeights(),
        class_weights: tuple[float, float, float, float] = (
            0.20,
            1.0,
            1.0,
            2.5,
        ),
    ) -> None:
        super().__init__()
        self.weights = weights
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(
        self,
        outputs: dict[str, object],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        classes = batch["classes"]
        full_target = batch["av"]
        av_target = full_target[:, (0, 2)]
        roi = batch["roi"]
        centerlines = batch["centerlines"]
        av_centerlines = centerlines[:, (0, 2)]

        av_logits = outputs["av_logits"]
        if not isinstance(av_logits, list):
            raise TypeError("outputs['av_logits'] must be a list")
        recurrent_weights = torch.linspace(
            0.35,
            1.0,
            len(av_logits),
            device=av_target.device,
            dtype=av_target.dtype,
        )
        recurrent_weights = recurrent_weights / recurrent_weights.sum()
        recurrent_av = av_target.new_zeros(())
        for iteration_weight, iteration_logits in zip(
            recurrent_weights,
            av_logits,
        ):
            probability = torch.sigmoid(iteration_logits)
            iteration_loss = (
                binary_focal_loss(
                    iteration_logits,
                    av_target,
                    roi,
                    alpha=0.86,
                    gamma=1.5,
                )
                + soft_dice_loss(probability, av_target, roi)
                + 0.55
                * focal_tversky_loss(
                    probability,
                    av_target,
                    roi,
                    alpha=0.25,
                    beta=0.75,
                    gamma=0.75,
                )
                + 0.40
                * conditional_av_loss(
                    iteration_logits,
                    av_target,
                    classes,
                )
            )
            recurrent_av = recurrent_av + iteration_weight * iteration_loss

        final_av_probability = torch.sigmoid(av_logits[-1])
        vessel_logits = outputs["vessel_logits"]
        semantic_logits = outputs["semantic_logits"]
        centerline_logits = outputs["centerline_logits"]
        aux_semantic_logits = outputs["aux_semantic_logits"]
        if not all(
            isinstance(value, Tensor)
            for value in (
                vessel_logits,
                semantic_logits,
                centerline_logits,
                aux_semantic_logits,
            )
        ):
            raise TypeError("all non-recurrent model outputs must be tensors")

        vessel_target = full_target[:, 1:2]
        vessel_probability = torch.sigmoid(vessel_logits)
        vessel = (
            binary_focal_loss(
                vessel_logits,
                vessel_target,
                roi,
                alpha=0.84,
                gamma=1.5,
            )
            + soft_dice_loss(vessel_probability, vessel_target, roi)
            + 0.60
            * focal_tversky_loss(
                vessel_probability,
                vessel_target,
                roi,
                alpha=0.22,
                beta=0.78,
                gamma=0.75,
            )
        )

        topology_coverage = centerline_coverage_loss(
            final_av_probability,
            av_centerlines,
            roi,
        )
        topology_cldice = soft_cldice_loss(
            final_av_probability,
            av_target,
            av_centerlines,
            roi,
            iterations=8,
            downsample=2,
        )
        topology = topology_coverage + 0.50 * topology_cldice
        hard_gap = hard_centerline_gap_loss(
            final_av_probability,
            av_centerlines,
            roi,
        )

        centerline_probability = torch.sigmoid(centerline_logits)
        centerline = binary_focal_loss(
            centerline_logits,
            centerlines,
            roi,
            alpha=0.94,
            gamma=1.5,
        ) + soft_dice_loss(
            centerline_probability,
            centerlines,
            roi,
            channel_weights=(1.0, 0.8, 1.0),
        )

        semantic = focal_cross_entropy(
            semantic_logits,
            classes,
            roi,
            self.class_weights,
            gamma=1.5,
        )

        aux_size = aux_semantic_logits.shape[-2:]
        aux_classes = F.interpolate(
            classes[:, None].float(),
            size=aux_size,
            mode="nearest",
        )[:, 0].long()
        aux_roi = F.interpolate(roi, size=aux_size, mode="nearest")
        aux_target = F.interpolate(
            full_target,
            size=aux_size,
            mode="nearest",
        )
        _, aux_probability = semantic_probabilities(aux_semantic_logits)
        deep_supervision = focal_cross_entropy(
            aux_semantic_logits,
            aux_classes,
            aux_roi,
            self.class_weights,
            gamma=1.5,
        ) + soft_dice_loss(
            aux_probability,
            aux_target,
            aux_roi,
            channel_weights=(1.0, 1.2, 1.0),
        )

        semantic_probability = torch.softmax(semantic_logits, dim=1)
        semantic_vessel = 1.0 - semantic_probability[:, :1]
        structured_vessel = torch.maximum(semantic_vessel, vessel_probability)
        hierarchy = (
            F.relu(final_av_probability - structured_vessel).square() * roi
        ).mean()

        total = (
            self.weights.recurrent_av * recurrent_av
            + self.weights.vessel * vessel
            + self.weights.topology * topology
            + self.weights.hard_gap * hard_gap
            + self.weights.centerline * centerline
            + self.weights.semantic * semantic
            + self.weights.deep_supervision * deep_supervision
            + self.weights.hierarchy * hierarchy
        )
        return {
            "total": total,
            "recurrent_av": recurrent_av,
            "vessel": vessel,
            "topology": topology,
            "topology_coverage": topology_coverage,
            "topology_cldice": topology_cldice,
            "hard_gap": hard_gap,
            "centerline": centerline,
            "semantic": semantic,
            "deep_supervision": deep_supervision,
            "hierarchy": hierarchy,
        }
