#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gave2v1.path_topology import topology_path_counts


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _binary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    truth_values = truth[mask]
    prediction_values = prediction[mask]
    tp = int((truth_values & prediction_values).sum())
    tn = int(((~truth_values) & (~prediction_values)).sum())
    fp = int(((~truth_values) & prediction_values).sum())
    fn = int((truth_values & (~prediction_values)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, specificity, accuracy


def _dice(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, int]:
    truth_values = truth & mask
    prediction_values = prediction & mask
    return (
        int((truth_values & prediction_values).sum()),
        int(truth_values.sum()),
        int(prediction_values.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GAVE2 A/V predictions with the RRWNet path metric."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    labels_dir = Path(args.labels_dir)
    masks_dir = Path(args.masks_dir)
    prediction_paths = sorted(predictions_dir.glob("g_*.png"))
    if not prediction_paths:
        raise FileNotFoundError(f"No g_*.png files in {predictions_dir}")

    class_truth: list[list[np.ndarray]] = [[], []]
    class_prediction: list[list[np.ndarray]] = [[], []]
    topology_rows: list[dict[str, float | str]] = []
    vessel_intersection = 0
    vessel_truth_sum = 0
    vessel_prediction_sum = 0

    for case_index, prediction_path in enumerate(prediction_paths):
        case_id = prediction_path.stem
        prediction_rgb = _read_rgb(prediction_path)
        label_rgb = _read_rgb(labels_dir / f"{case_id}.png")
        roi = _read_mask(masks_dir / f"{case_id}.png")
        if prediction_rgb.shape != label_rgb.shape:
            raise ValueError(
                f"{case_id}: prediction {prediction_rgb.shape} != label {label_rgb.shape}"
            )

        probability_artery = prediction_rgb[..., 0].astype(np.float32) / 255.0
        probability_vessel = prediction_rgb[..., 1].astype(np.float32) / 255.0
        probability_vein = prediction_rgb[..., 2].astype(np.float32) / 255.0
        predicted_artery = probability_artery > 0.5
        predicted_vessel = probability_vessel > 0.5
        predicted_vein = probability_vein > 0.5

        red = label_rgb[..., 0] > 127
        green = label_rgb[..., 1] > 127
        blue = label_rgb[..., 2] > 127
        ground_artery = (red | green) & roi
        ground_vein = (blue | green) & roi
        ground_vessel = (red | green | blue) & roi
        # GAVE2 evaluates each A/V map as an independent binary segmentation
        # over all valid ROI pixels. This is confirmed by the public
        # leaderboard's background-dominated A/V accuracies (~0.98).
        class_truth[0].append(ground_artery[roi])
        class_truth[1].append(ground_vein[roi])
        class_prediction[0].append(predicted_artery[roi])
        class_prediction[1].append(predicted_vein[roi])

        intersection, truth_sum, prediction_sum = _dice(
            ground_vessel,
            predicted_vessel,
            roi,
        )
        vessel_intersection += intersection
        vessel_truth_sum += truth_sum
        vessel_prediction_sum += prediction_sum

        topology_values: dict[str, float | str] = {"id": case_id}
        for class_index, (name, truth, probability) in enumerate(
            (
                ("artery", ground_artery, probability_artery * roi),
                ("vein", ground_vein, probability_vein * roi),
            )
        ):
            infeasible, _, correct = topology_path_counts(
                truth,
                probability,
                threshold=0.5,
                n_paths=args.n_paths,
                seed=args.seed + 2 * case_index + class_index,
            )
            topology_values[f"{name}_inf"] = infeasible / args.n_paths
            topology_values[f"{name}_cor"] = correct / args.n_paths
        topology_rows.append(topology_values)
        print(json.dumps(topology_values), flush=True)

    classification: dict[str, dict[str, float]] = {}
    for class_index, name in enumerate(("artery", "vein")):
        sensitivity, specificity, accuracy = _binary_metrics(
            np.concatenate(class_truth[class_index]),
            np.concatenate(class_prediction[class_index]),
            np.ones(
                sum(len(values) for values in class_truth[class_index]),
                dtype=bool,
            ),
        )
        classification[name] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
        }

    dice = (
        2.0 * vessel_intersection / (vessel_truth_sum + vessel_prediction_sum)
        if vessel_truth_sum + vessel_prediction_sum
        else 0.0
    )
    artery_inf = float(np.mean([row["artery_inf"] for row in topology_rows]))
    artery_cor = float(np.mean([row["artery_cor"] for row in topology_rows]))
    vein_inf = float(np.mean([row["vein_inf"] for row in topology_rows]))
    vein_cor = float(np.mean([row["vein_cor"] for row in topology_rows]))

    class_components = [
        0.3 * values["sensitivity"]
        + 0.3 * values["specificity"]
        + 0.4 * values["accuracy"]
        for values in classification.values()
    ]
    topology_components = [
        0.5 * artery_cor + 0.5 * (1.0 - artery_inf),
        0.5 * vein_cor + 0.5 * (1.0 - vein_inf),
    ]
    classification_component = float(np.mean(class_components))
    topology_component = float(np.mean(topology_components))
    documented_score = 10.0 * (
        0.3 * classification_component
        + 0.4 * dice
        + 0.3 * topology_component
    )
    leaderboard_reconstructed_score = 10.0 * (
        0.4 * classification_component
        + 0.2 * dice
        + 0.4 * topology_component
    )
    result = {
        "n_cases": len(prediction_paths),
        "n_paths_per_case_and_class": args.n_paths,
        "threshold": 0.5,
        "classification": classification,
        "av_dsc": dice,
        "topology": {
            "artery_inf": artery_inf,
            "artery_cor": artery_cor,
            "vein_inf": vein_inf,
            "vein_cor": vein_cor,
        },
        "classification_component": classification_component,
        "topology_component": topology_component,
        "documented_task_score": documented_score,
        "leaderboard_reconstructed_task_score": leaderboard_reconstructed_score,
        "task_score": leaderboard_reconstructed_score,
        "task_score_rule": "leaderboard_reconstructed",
        "score_weights": {
            "documented": {
                "classification": 0.3,
                "dsc": 0.4,
                "topology": 0.3,
            },
            "leaderboard_reconstructed": {
                "classification": 0.4,
                "dsc": 0.2,
                "topology": 0.4,
            },
        },
        "per_case_topology": topology_rows,
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
