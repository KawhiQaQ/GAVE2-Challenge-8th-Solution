#!/usr/bin/env python3
"""Evaluate the released GAVE2 biomarker extractor on local labeled data.

Two segmentation sources are supported:

* ``ground_truth``: an oracle check using the provided A/V annotations.
* A directory of RGB probability PNGs in challenge order [A, Vessel, V].

The script uses the released CMRRWNet biomarker formulas verbatim after turning
probability maps into binary artery/vein masks. It reports raw MAE and SMAPE;
the hidden challenge P/Q normalization cannot be reconstructed from public
information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from get_biomarker import (
    calculate_crae_crve_revised,
    calculate_density_in_c,
    generate_annular_masks,
    get_od_max_circle,
    get_top_n_vessels_in_c,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "GAVE2_preliminary"
EVALUATED_KEYS = (
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)
ALL_KEYS = ("CRAE", "CRVE", *EVALUATED_KEYS)


def parse_biomarker(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()[:2]
        values[key] = float(value)
    return values


def ground_truth_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    red = raw[..., 0] > 127
    green = raw[..., 1] > 127
    blue = raw[..., 2] > 127
    artery = (red | green).astype(np.uint8) * 255
    vein = (blue | green).astype(np.uint8) * 255
    return artery, vein


def prediction_masks(
    path: Path,
    threshold: float,
    mask_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    probability = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    probability = probability.astype(np.float32) / 255.0
    if mask_policy == "vessel_argmax":
        vessel = probability[..., 1] >= threshold
        artery_class = probability[..., 0] >= probability[..., 2]
        artery = (vessel & artery_class).astype(np.uint8) * 255
        vein = (vessel & (~artery_class)).astype(np.uint8) * 255
    elif mask_policy == "exclusive_av":
        vessel = probability[..., 1] >= threshold
        artery_high = probability[..., 0] >= threshold
        vein_high = probability[..., 2] >= threshold
        artery = (
            vessel & artery_high & (~vein_high)
        ).astype(np.uint8) * 255
        vein = (
            vessel & vein_high & (~artery_high)
        ).astype(np.uint8) * 255
    elif mask_policy == "independent_av":
        artery = (probability[..., 0] >= threshold).astype(np.uint8) * 255
        vein = (probability[..., 2] >= threshold).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown mask policy: {mask_policy}")
    return artery, vein


def fast_fractal_dimension(binary_image: np.ndarray) -> float:
    """Vectorized equivalent of the released all-integer-box-size loop."""
    if binary_image.max() == 0:
        return 0.0
    binary = (binary_image > 127).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    rows, cols = skeleton.shape
    integral = cv2.integral(skeleton)
    log_inverse_sizes: list[float] = []
    log_counts: list[float] = []
    for box_size in range(1, min(rows, cols) // 2 + 1):
        row_starts = np.arange(0, rows, box_size)
        col_starts = np.arange(0, cols, box_size)
        row_ends = np.minimum(row_starts + box_size, rows)
        col_ends = np.minimum(col_starts + box_size, cols)
        sums = (
            integral[np.ix_(row_ends, col_ends)]
            - integral[np.ix_(row_starts, col_ends)]
            - integral[np.ix_(row_ends, col_starts)]
            + integral[np.ix_(row_starts, col_starts)]
        )
        count = int(np.count_nonzero(sums))
        if count > 0:
            log_inverse_sizes.append(float(np.log(1.0 / box_size)))
            log_counts.append(float(np.log(count)))
    if len(log_inverse_sizes) < 2:
        return 0.0
    return float(np.polyfit(log_inverse_sizes, log_counts, 1)[0])


def calculate_one(
    artery: np.ndarray,
    vein: np.ndarray,
    disc_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    _, od_bin = cv2.threshold(disc_mask, 200, 255, cv2.THRESH_BINARY)
    od_center, dd = get_od_max_circle(od_bin)
    if dd <= 0:
        raise ValueError("No optic disc component")
    dummy_rgb = np.zeros((*artery.shape, 3), dtype=np.uint8)
    _, _, c_mask = generate_annular_masks(dummy_rgb, od_center, dd)
    artery_widths = get_top_n_vessels_in_c(artery, c_mask, top_n=6)
    vein_widths = get_top_n_vessels_in_c(vein, c_mask, top_n=6)
    crae = calculate_crae_crve_revised(artery_widths, is_artery=True)
    crve = calculate_crae_crve_revised(vein_widths, is_artery=False)
    values = {
        "CRAE": float(crae),
        "CRVE": float(crve),
        "AVR": float(crae / crve) if crae > 0 and crve > 0 else 0.0,
        "artery_density": float(calculate_density_in_c(artery, c_mask)),
        "vein_density": float(calculate_density_in_c(vein, c_mask)),
        "artery_fractal_dimension": fast_fractal_dimension(artery),
        "vein_fractal_dimension": fast_fractal_dimension(vein),
    }
    diagnostics = {
        "disc_center_x": float(od_center[0]),
        "disc_center_y": float(od_center[1]),
        "disc_diameter": float(dd),
        "artery_components_used": float(sum(width > 0 for width in artery_widths)),
        "vein_components_used": float(sum(width > 0 for width in vein_widths)),
    }
    return values, diagnostics


def smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = (np.abs(truth) + np.abs(prediction)) / 2.0
    terms = np.divide(
        np.abs(truth - prediction),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(100.0 * terms.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--disc-dir", required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument(
        "--source",
        default="ground_truth",
        help="'ground_truth' or a directory containing probability PNGs.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--mask-policy",
        choices=("vessel_argmax", "exclusive_av", "independent_av"),
        default="vessel_argmax",
        help=(
            "How probability PNGs become Task3 artery/vein masks. "
            "'vessel_argmax' preserves the historical V1/V2 pipeline; "
            "'exclusive_av' mirrors released get_biomarker.py after "
            "thresholding [A, vessel, V] channels; "
            "'independent_av' follows the submitted R/B channel semantics."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    disc_dir = Path(args.disc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = None if args.source == "ground_truth" else Path(args.source)
    case_ids = sorted(
        path.stem
        for path in (
            (data_root / args.split / "images")
            if source_dir is None
            else source_dir
        ).glob("g_*.png")
    )
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    truth_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, float]] = []
    per_case: dict[str, dict[str, object]] = {}
    if source_dir is None and args.split != "training":
        raise ValueError("ground_truth source is only available for the training split")

    for case_id in case_ids:
        if source_dir is None:
            artery, vein = ground_truth_masks(data_root / "training" / "av" / f"{case_id}.png")
        else:
            artery, vein = prediction_masks(
                source_dir / f"{case_id}.png",
                args.threshold,
                args.mask_policy,
            )
        disc = cv2.imread(str(disc_dir / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
        if disc is None:
            raise FileNotFoundError(disc_dir / f"{case_id}.png")
        predicted, diagnostics = calculate_one(artery, vein, disc)
        prediction_rows.append(predicted)
        case_result: dict[str, object] = {
            "prediction": predicted,
            "diagnostics": diagnostics,
        }
        if args.split == "training":
            truth = parse_biomarker(data_root / "training" / "biomarker" / f"{case_id}.txt")
            truth_rows.append(truth)
            case_result["truth"] = truth
        per_case[case_id] = case_result
        with (output_dir / f"{case_id}.txt").open("w", encoding="utf-8") as handle:
            for key in ALL_KEYS:
                handle.write(f"{key} {predicted[key]:.6f}\n")
        print(case_id, json.dumps(predicted), flush=True)

    summary: dict[str, object] = {
        "n": len(case_ids),
        "split": args.split,
        "source": args.source,
        "mask_policy": args.mask_policy if source_dir is not None else "ground_truth",
        "metrics": {},
    }
    if args.split == "training":
        for key in ALL_KEYS:
            truth = np.asarray([row[key] for row in truth_rows], dtype=np.float64)
            prediction = np.asarray([row[key] for row in prediction_rows], dtype=np.float64)
            summary["metrics"][key] = {
                "mae": float(np.abs(truth - prediction).mean()),
                "smape_percent": smape(truth, prediction),
                "truth_mean": float(truth.mean()),
                "prediction_mean": float(prediction.mean()),
            }
        summary["evaluated_macro_mae"] = float(
            np.mean([summary["metrics"][key]["mae"] for key in EVALUATED_KEYS])
        )
        summary["evaluated_macro_smape_percent"] = float(
            np.mean([summary["metrics"][key]["smape_percent"] for key in EVALUATED_KEYS])
        )
    (output_dir / "per_case.json").write_text(json.dumps(per_case, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
