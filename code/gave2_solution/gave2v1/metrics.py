from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.morphology import skeletonize
from torch import Tensor


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def topology_cldice(
    probability: np.ndarray,
    target: np.ndarray,
    target_centerline: np.ndarray,
    threshold: float,
) -> float:
    prediction = probability >= threshold
    predicted_centerline = skeletonize(prediction)
    topology_precision = _safe_ratio(
        float((predicted_centerline & target).sum()),
        float(predicted_centerline.sum()),
    )
    topology_sensitivity = _safe_ratio(
        float((target_centerline & prediction).sum()),
        float(target_centerline.sum()),
    )
    return _safe_ratio(
        2.0 * topology_precision * topology_sensitivity,
        topology_precision + topology_sensitivity,
    )


@dataclass
class MetricAccumulator:
    threshold: float = 0.5
    intersections: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    prediction_sums: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_sums: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    topology_values: list[list[float]] = field(default_factory=list)
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)

    def update(
        self,
        case_id: str,
        probabilities: Tensor,
        target: Tensor,
        roi: Tensor,
        classes: Tensor,
        centerlines: Tensor,
    ) -> None:
        probability = probabilities.detach().float().cpu().numpy()
        truth = target.detach().bool().cpu().numpy()
        valid_roi = roi.detach().bool().cpu().numpy()[0]
        class_map = classes.detach().cpu().numpy()
        target_centerlines = centerlines.detach().bool().cpu().numpy()
        binary = probability >= self.threshold
        case_dice = []
        for channel in range(3):
            predicted = binary[channel] & valid_roi
            ground_truth = truth[channel] & valid_roi
            intersection = float((predicted & ground_truth).sum())
            prediction_sum = float(predicted.sum())
            target_sum = float(ground_truth.sum())
            self.intersections[channel] += intersection
            self.prediction_sums[channel] += prediction_sum
            self.target_sums[channel] += target_sum
            case_dice.append(
                _safe_ratio(2.0 * intersection, prediction_sum + target_sum)
            )

        classification_mask = valid_roi & ((class_map == 1) | (class_map == 2))
        ground_truth_artery = class_map[classification_mask] == 1
        predicted_artery = (
            probability[0][classification_mask] >= probability[2][classification_mask]
        )
        self.tp += int((predicted_artery & ground_truth_artery).sum())
        self.tn += int(((~predicted_artery) & (~ground_truth_artery)).sum())
        self.fp += int((predicted_artery & (~ground_truth_artery)).sum())
        self.fn += int(((~predicted_artery) & ground_truth_artery).sum())

        topology = [
            topology_cldice(
                probability[channel] * valid_roi,
                truth[channel] & valid_roi,
                target_centerlines[channel] & valid_roi,
                self.threshold,
            )
            for channel in range(3)
        ]
        self.topology_values.append(topology)
        self.per_case[case_id] = {
            "dice_artery": case_dice[0],
            "dice_vessel": case_dice[1],
            "dice_vein": case_dice[2],
            "cldice_artery": topology[0],
            "cldice_vessel": topology[1],
            "cldice_vein": topology[2],
        }

    def summarize(self) -> dict[str, float | dict[str, dict[str, float]]]:
        dice = [
            _safe_ratio(
                2.0 * self.intersections[channel],
                self.prediction_sums[channel] + self.target_sums[channel],
            )
            for channel in range(3)
        ]
        sensitivity = _safe_ratio(self.tp, self.tp + self.fn)
        specificity = _safe_ratio(self.tn, self.tn + self.fp)
        accuracy = _safe_ratio(
            self.tp + self.tn,
            self.tp + self.tn + self.fp + self.fn,
        )
        classification = (
            0.3 * sensitivity + 0.3 * specificity + 0.4 * accuracy
        )
        if self.topology_values:
            topology_array = np.asarray(self.topology_values, dtype=np.float64)
            topology_channels = topology_array.mean(axis=0)
        else:
            topology_channels = np.zeros(3, dtype=np.float64)
        topology_proxy = float(topology_channels.mean())
        score_proxy = 10.0 * (
            0.4 * dice[1] + 0.3 * classification + 0.3 * topology_proxy
        )
        return {
            "threshold": self.threshold,
            "dice_artery": dice[0],
            "dice_vessel": dice[1],
            "dice_vein": dice[2],
            "sensitivity_av": sensitivity,
            "specificity_av": specificity,
            "accuracy_av": accuracy,
            "classification_component": classification,
            "cldice_artery": float(topology_channels[0]),
            "cldice_vessel": float(topology_channels[1]),
            "cldice_vein": float(topology_channels[2]),
            "topology_proxy": topology_proxy,
            "task_score_proxy": score_proxy,
            "per_case": self.per_case,
        }


def sweep_thresholds(
    samples: dict[str, dict[str, Tensor]],
    thresholds: np.ndarray | None = None,
) -> dict[str, object]:
    if thresholds is None:
        thresholds = np.linspace(0.15, 0.75, 25)
    rows = []
    for threshold in thresholds:
        accumulator = MetricAccumulator(threshold=float(threshold))
        for case_id, sample in samples.items():
            accumulator.update(
                case_id,
                sample["probabilities"],
                sample["av"],
                sample["roi"],
                sample["classes"],
                sample["centerlines"],
            )
        summary = accumulator.summarize()
        summary.pop("per_case", None)
        rows.append(summary)
    best_dice = max(rows, key=lambda row: row["dice_vessel"])
    best_proxy = max(rows, key=lambda row: row["task_score_proxy"])
    return {
        "best_vessel_dice": best_dice,
        "best_score_proxy": best_proxy,
        "rows": rows,
    }
