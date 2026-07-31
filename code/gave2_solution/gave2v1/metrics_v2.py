from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.measure import label
from torch import Tensor


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def expected_path_feasibility(
    ground_centerline: np.ndarray,
    predicted_mask: np.ndarray,
) -> float:
    """Expected RRWNet path feasibility without Monte Carlo shortest paths."""
    ground_components = label(ground_centerline.astype(bool))
    predicted_components = label(predicted_mask.astype(bool))
    total_ground = int((ground_components > 0).sum())
    if total_ground == 0:
        return 0.0

    expected = 0.0
    for ground_id in range(1, int(ground_components.max()) + 1):
        component_mask = ground_components == ground_id
        component_size = int(component_mask.sum())
        if component_size == 0:
            continue
        predicted_ids, counts = np.unique(
            predicted_components[component_mask],
            return_counts=True,
        )
        non_background_counts = counts[predicted_ids != 0].astype(np.float64)
        conditional = float(
            np.square(non_background_counts).sum() / (component_size**2)
        )
        expected += component_size / total_ground * conditional
    return expected


@dataclass
class V2MetricAccumulator:
    threshold: float = 0.5
    intersections: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    prediction_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    target_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 4), dtype=np.int64)
    )
    feasibility_values: list[list[float]] = field(default_factory=list)
    centerline_recall_values: list[list[float]] = field(default_factory=list)
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
        binary = probability > self.threshold

        case_dice: list[float] = []
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

        classification_mask = valid_roi
        for metric_index, channel in enumerate((0, 2)):
            ground_truth = truth[channel][classification_mask]
            predicted = binary[channel][classification_mask]
            tp = int((ground_truth & predicted).sum())
            tn = int(((~ground_truth) & (~predicted)).sum())
            fp = int(((~ground_truth) & predicted).sum())
            fn = int((ground_truth & (~predicted)).sum())
            self.confusion[metric_index] += (tp, tn, fp, fn)

        feasibility: list[float] = []
        centerline_recall: list[float] = []
        for channel in (0, 2):
            ground_centerline = target_centerlines[channel] & valid_roi
            predicted = binary[channel] & valid_roi
            feasibility.append(
                expected_path_feasibility(ground_centerline, predicted)
            )
            centerline_recall.append(
                _safe_ratio(
                    float((ground_centerline & predicted).sum()),
                    float(ground_centerline.sum()),
                )
            )
        self.feasibility_values.append(feasibility)
        self.centerline_recall_values.append(centerline_recall)
        self.per_case[case_id] = {
            "dice_artery": case_dice[0],
            "dice_vessel": case_dice[1],
            "dice_vein": case_dice[2],
            "feasibility_artery": feasibility[0],
            "feasibility_vein": feasibility[1],
            "centerline_recall_artery": centerline_recall[0],
            "centerline_recall_vein": centerline_recall[1],
        }

    def summarize(self) -> dict[str, object]:
        dice = [
            _safe_ratio(
                2.0 * self.intersections[channel],
                self.prediction_sums[channel] + self.target_sums[channel],
            )
            for channel in range(3)
        ]
        classification_rows: list[dict[str, float]] = []
        classification_components: list[float] = []
        for tp, tn, fp, fn in self.confusion:
            sensitivity = _safe_ratio(float(tp), float(tp + fn))
            specificity = _safe_ratio(float(tn), float(tn + fp))
            accuracy = _safe_ratio(float(tp + tn), float(tp + tn + fp + fn))
            classification_rows.append(
                {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "accuracy": accuracy,
                }
            )
            classification_components.append(
                0.3 * sensitivity + 0.3 * specificity + 0.4 * accuracy
            )

        if self.feasibility_values:
            feasibility = np.asarray(self.feasibility_values).mean(axis=0)
            centerline_recall = np.asarray(self.centerline_recall_values).mean(axis=0)
        else:
            feasibility = np.zeros(2, dtype=np.float64)
            centerline_recall = np.zeros(2, dtype=np.float64)
        classification_component = float(np.mean(classification_components))
        topology_proxy = float(np.mean(feasibility))
        score_proxy = 10.0 * (
            0.4 * classification_component
            + 0.2 * dice[1]
            + 0.4 * topology_proxy
        )
        return {
            "threshold": self.threshold,
            "dice_artery": dice[0],
            "dice_vessel": dice[1],
            "dice_vein": dice[2],
            "artery": classification_rows[0],
            "vein": classification_rows[1],
            "classification_component": classification_component,
            "feasibility_artery": float(feasibility[0]),
            "feasibility_vein": float(feasibility[1]),
            "topology_proxy": topology_proxy,
            "centerline_recall_artery": float(centerline_recall[0]),
            "centerline_recall_vein": float(centerline_recall[1]),
            "task_score_proxy": score_proxy,
            "per_case": self.per_case,
        }
