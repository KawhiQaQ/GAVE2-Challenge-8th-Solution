from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize


TASK3_EVALUATED_KEYS = (
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def _smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = (np.abs(truth) + np.abs(prediction)) / 2.0
    values = np.divide(
        np.abs(truth - prediction),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(100.0 * values.mean())


def summarize_task3(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, Any]:
    prediction = np.concatenate(predictions, axis=0).astype(np.float64)
    truth = np.concatenate(targets, axis=0).astype(np.float64)
    metrics: dict[str, dict[str, float]] = {}
    normalized_errors: list[float] = []
    for index, key in enumerate(TASK3_EVALUATED_KEYS):
        mae = float(np.abs(truth[:, index] - prediction[:, index]).mean())
        smape = _smape(truth[:, index], prediction[:, index])
        truth_scale = float(np.abs(truth[:, index]).mean())
        relative_mae = mae / max(truth_scale, 1e-8)
        normalized = 0.5 * relative_mae + 0.5 * smape / 100.0
        metrics[key] = {
            "mae": mae,
            "smape_percent": smape,
            "truth_mean": float(truth[:, index].mean()),
            "prediction_mean": float(prediction[:, index].mean()),
            "selection_error": normalized,
        }
        normalized_errors.append(normalized)
    return {
        "metrics": metrics,
        "selection_error": float(np.mean(normalized_errors)),
        "selection_rule": (
            "mean over five exact biomarkers of "
            "0.5*(MAE/mean_abs_truth)+0.5*(SMAPE/100)"
        ),
    }


def _disc_c_mask(disc_mask: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(disc_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No optic disc component")
    contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    diameter = 2.0 * radius
    center = (int(center_x), int(center_y))
    inner_radius = int(radius + diameter)
    outer_radius = int(radius + 2.0 * diameter)
    c_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    cv2.circle(c_mask, center, outer_radius, 255, -1)
    cv2.circle(c_mask, center, inner_radius, 0, -1)
    return c_mask


def _top_six_diameters(
    vessel_mask: np.ndarray,
    c_mask: np.ndarray,
) -> list[float]:
    vessel_in_c = cv2.bitwise_and(
        vessel_mask,
        vessel_mask,
        mask=c_mask,
    )
    _, binary = cv2.threshold(
        vessel_in_c,
        127,
        255,
        cv2.THRESH_BINARY,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        8,
        cv2.CV_32S,
    )
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        components.append(
            (
                -int(stats[index, cv2.CC_STAT_AREA]),
                int(stats[index, cv2.CC_STAT_LEFT]),
                int(stats[index, cv2.CC_STAT_TOP]),
                int(stats[index, cv2.CC_STAT_WIDTH]),
                int(stats[index, cv2.CC_STAT_HEIGHT]),
            )
        )
    diameters: list[float] = []
    for _, x, y, width, height in sorted(components)[:6]:
        component_roi = np.zeros_like(vessel_mask)
        component_roi[y : y + height, x : x + width] = 255
        vessel_roi = cv2.bitwise_and(vessel_mask, component_roi)
        skeleton, distance = medial_axis(vessel_roi, return_distance=True)
        values = distance[skeleton] * 2.0
        diameters.append(float(values.max()) if values.size else 0.0)
    return diameters + [0.0] * (6 - len(diameters))


def _combine_diameters(
    diameters: list[float],
    *,
    artery: bool,
) -> float:
    coefficient = 0.88 if artery else 0.95
    values = sorted(diameters, reverse=True)
    while len(values) > 1:
        values = sorted(values, reverse=True)
        next_values: list[float] = []
        left = 0
        right = len(values) - 1
        while left < right:
            next_values.append(
                coefficient
                * math.sqrt(values[left] ** 2 + values[right] ** 2)
            )
            left += 1
            right -= 1
        values = next_values
    return float(values[0]) if values else 0.0


def _density(mask: np.ndarray, c_mask: np.ndarray) -> float:
    vessel = (mask > 127).astype(np.uint8)
    region = (c_mask > 127).astype(np.uint8)
    denominator = int(region.sum())
    if denominator == 0:
        return 0.0
    return float((vessel * region).sum() / denominator)


def _fast_fractal_dimension(mask: np.ndarray) -> float:
    if mask.max() == 0:
        return 0.0
    skeleton = skeletonize(mask > 127).astype(np.uint8)
    rows, columns = skeleton.shape
    integral = cv2.integral(skeleton)
    inverse_sizes: list[float] = []
    counts: list[float] = []
    for box_size in range(1, min(rows, columns) // 2 + 1):
        row_starts = np.arange(0, rows, box_size)
        column_starts = np.arange(0, columns, box_size)
        row_ends = np.minimum(row_starts + box_size, rows)
        column_ends = np.minimum(column_starts + box_size, columns)
        sums = (
            integral[np.ix_(row_ends, column_ends)]
            - integral[np.ix_(row_starts, column_ends)]
            - integral[np.ix_(row_ends, column_starts)]
            + integral[np.ix_(row_starts, column_starts)]
        )
        occupied = int(np.count_nonzero(sums))
        if occupied > 0:
            inverse_sizes.append(float(np.log(1.0 / box_size)))
            counts.append(float(np.log(occupied)))
    if len(inverse_sizes) < 2:
        return 0.0
    return float(np.polyfit(inverse_sizes, counts, 1)[0])


def _quantized_masks(
    probability: np.ndarray,
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if probability.shape[0] != 3:
        raise ValueError("Expected [artery, vessel, vein] probabilities")
    quantized = np.rint(
        np.clip(probability, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    artery_probability, vessel_probability, vein_probability = quantized
    if policy == "independent_av":
        artery = artery_probability >= 128
        vein = vein_probability >= 128
    elif policy == "vessel_argmax":
        vessel = vessel_probability >= 128
        artery_class = artery_probability >= vein_probability
        artery = vessel & artery_class
        vein = vessel & (~artery_class)
    else:
        raise ValueError(f"Unknown mask policy: {policy}")
    return (
        artery.astype(np.uint8) * 255,
        vein.astype(np.uint8) * 255,
    )


class ExactTask3Calculator:
    """Exact released biomarker geometry with V23-style fixed source routing."""

    def __init__(self, disc_dir: str | Path) -> None:
        self.disc_dir = Path(disc_dir)
        self.c_masks: dict[str, np.ndarray] = {}

    def _c_mask(self, case_id: str) -> np.ndarray:
        if case_id not in self.c_masks:
            path = self.disc_dir / f"{case_id}.png"
            disc = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if disc is None:
                raise FileNotFoundError(path)
            self.c_masks[case_id] = _disc_c_mask(disc)
        return self.c_masks[case_id]

    def calculate(
        self,
        case_id: str,
        dense_probability: np.ndarray,
        geometry_probability: np.ndarray,
    ) -> np.ndarray:
        c_mask = self._c_mask(case_id)
        caliber_artery, caliber_vein = _quantized_masks(
            dense_probability,
            policy="independent_av",
        )
        dense_artery, dense_vein = _quantized_masks(
            dense_probability,
            policy="vessel_argmax",
        )
        geometry_artery, geometry_vein = _quantized_masks(
            geometry_probability,
            policy="vessel_argmax",
        )
        crae = _combine_diameters(
            _top_six_diameters(caliber_artery, c_mask),
            artery=True,
        )
        crve = _combine_diameters(
            _top_six_diameters(caliber_vein, c_mask),
            artery=False,
        )
        return np.asarray(
            [
                crae / crve if crae > 0 and crve > 0 else 0.0,
                _density(dense_artery, c_mask),
                _density(dense_vein, c_mask),
                _fast_fractal_dimension(geometry_artery),
                _fast_fractal_dimension(geometry_vein),
            ],
            dtype=np.float64,
        )
