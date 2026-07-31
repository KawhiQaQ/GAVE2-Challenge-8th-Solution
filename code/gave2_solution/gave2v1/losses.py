from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def semantic_probabilities(logits: Tensor) -> tuple[Tensor, Tensor]:
    semantic = torch.softmax(logits, dim=1)
    background, artery, vein, overlap = semantic.unbind(dim=1)
    av = torch.stack((artery + overlap, 1.0 - background, vein + overlap), dim=1)
    return semantic, av


def focal_cross_entropy(
    logits: Tensor,
    target: Tensor,
    roi: Tensor,
    class_weights: Tensor,
    gamma: float = 2.0,
) -> Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    target_index = target.unsqueeze(1)
    log_pt = log_probabilities.gather(1, target_index).squeeze(1)
    pt = probabilities.gather(1, target_index).squeeze(1)
    weights = class_weights[target]
    loss = -weights * (1.0 - pt).pow(gamma) * log_pt
    return masked_mean(loss, roi[:, 0])


def soft_dice_loss(
    probabilities: Tensor,
    target: Tensor,
    roi: Tensor,
    channel_weights: tuple[float, ...] | None = None,
    smooth: float = 1.0,
) -> Tensor:
    roi = roi.to(probabilities.dtype)
    probabilities = probabilities * roi
    target = target.to(probabilities.dtype) * roi
    dims = (0, 2, 3)
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    losses = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    if channel_weights is None:
        return losses.mean()
    weights = probabilities.new_tensor(channel_weights)
    return (losses * weights).sum() / weights.sum()


def focal_tversky_loss(
    probabilities: Tensor,
    target: Tensor,
    roi: Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1.0,
) -> Tensor:
    roi = roi.to(probabilities.dtype)
    probabilities = probabilities * roi
    target = target.to(probabilities.dtype) * roi
    dims = (0, 2, 3)
    true_positive = (probabilities * target).sum(dim=dims)
    false_positive = (probabilities * (1.0 - target) * roi).sum(dim=dims)
    false_negative = ((1.0 - probabilities) * target * roi).sum(dim=dims)
    tversky = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - tversky).pow(gamma).mean()


def binary_focal_loss(
    logits: Tensor,
    target: Tensor,
    roi: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    target = target.to(logits.dtype)
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(alpha_t * (1.0 - pt).pow(gamma) * bce, roi)


def soft_erode(image: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(image: Tensor) -> Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: Tensor) -> Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: Tensor, iterations: int = 5) -> Tensor:
    image = image.clamp(0.0, 1.0)
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probabilities: Tensor,
    target: Tensor,
    centerlines: Tensor,
    roi: Tensor,
    iterations: int = 5,
    downsample: int = 2,
    smooth: float = 1.0,
) -> Tensor:
    if downsample > 1:
        probabilities = F.avg_pool2d(probabilities, downsample)
        target = F.max_pool2d(target, downsample)
        centerlines = F.max_pool2d(centerlines, downsample)
        roi = F.max_pool2d(roi, downsample)
    probabilities = probabilities * roi
    target = target * roi
    centerlines = centerlines * roi
    predicted_skeleton = soft_skeletonize(probabilities, iterations=iterations)
    dims = (0, 2, 3)
    topology_precision = (predicted_skeleton * target).sum(dim=dims)
    topology_precision = (topology_precision + smooth) / (
        predicted_skeleton.sum(dim=dims) + smooth
    )
    topology_sensitivity = (centerlines * probabilities).sum(dim=dims)
    topology_sensitivity = (topology_sensitivity + smooth) / (
        centerlines.sum(dim=dims) + smooth
    )
    cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + 1e-6)
    )
    return 1.0 - cldice.mean()


@dataclass(frozen=True)
class LossWeights:
    semantic: float = 1.0
    dice: float = 1.0
    tversky: float = 0.45
    vessel_aux: float = 0.45
    centerline: float = 0.35
    cldice: float = 0.40
    deep_supervision: float = 0.30


class GAVEV1Loss(nn.Module):
    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        class_weights: tuple[float, float, float, float] = (0.20, 1.0, 1.0, 2.5),
    ) -> None:
        super().__init__()
        self.weights = weights
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        classes = batch["classes"]
        av_target = batch["av"]
        roi = batch["roi"]
        centerlines = batch["centerlines"]
        _, av_probabilities = semantic_probabilities(outputs["semantic_logits"])

        semantic = focal_cross_entropy(
            outputs["semantic_logits"],
            classes,
            roi,
            self.class_weights,
        )
        dice = soft_dice_loss(
            av_probabilities,
            av_target,
            roi,
            channel_weights=(1.0, 1.25, 1.0),
        )
        tversky = focal_tversky_loss(av_probabilities, av_target, roi)

        vessel_logits = outputs["vessel_logits"]
        vessel_target = av_target[:, 1:2]
        vessel_probability = torch.sigmoid(vessel_logits)
        vessel_aux = binary_focal_loss(
            vessel_logits, vessel_target, roi, alpha=0.8
        ) + soft_dice_loss(vessel_probability, vessel_target, roi)

        centerline_logits = outputs["centerline_logits"]
        centerline = binary_focal_loss(
            centerline_logits, centerlines, roi, alpha=0.92
        ) + soft_dice_loss(
            torch.sigmoid(centerline_logits),
            centerlines,
            roi,
            channel_weights=(1.0, 1.2, 1.0),
        )
        cldice = soft_cldice_loss(
            av_probabilities,
            av_target,
            centerlines,
            roi,
            iterations=5,
            downsample=2,
        )

        aux_logits = outputs["aux_semantic_logits"]
        aux_size = aux_logits.shape[-2:]
        aux_classes = F.interpolate(
            classes[:, None].float(), size=aux_size, mode="nearest"
        )[:, 0].long()
        aux_roi = F.interpolate(roi, size=aux_size, mode="nearest")
        aux_target = F.interpolate(av_target, size=aux_size, mode="nearest")
        _, aux_av = semantic_probabilities(aux_logits)
        deep_supervision = focal_cross_entropy(
            aux_logits,
            aux_classes,
            aux_roi,
            self.class_weights,
        ) + soft_dice_loss(aux_av, aux_target, aux_roi)

        total = (
            self.weights.semantic * semantic
            + self.weights.dice * dice
            + self.weights.tversky * tversky
            + self.weights.vessel_aux * vessel_aux
            + self.weights.centerline * centerline
            + self.weights.cldice * cldice
            + self.weights.deep_supervision * deep_supervision
        )
        return {
            "total": total,
            "semantic": semantic,
            "dice": dice,
            "tversky": tversky,
            "vessel_aux": vessel_aux,
            "centerline": centerline,
            "cldice": cldice,
            "deep_supervision": deep_supervision,
        }
