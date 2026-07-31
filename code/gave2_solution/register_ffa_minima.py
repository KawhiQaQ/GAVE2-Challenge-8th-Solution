#!/usr/bin/env python3
"""Register each FFA phase into CFP coordinates with official MINIMA-LoFTR.

Early and late FFA are matched independently because eye motion between phases
can be substantial.  Labels are never read or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


EXPECTED_MINIMA_LOFTR_SHA256 = (
    "810d19773ff898ba04a68c99a3eff9c112210bf884214bd76aec885e83b0e257"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def robust_unit(image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = image[valid] if valid is not None and bool(valid.any()) else image.ravel()
    values = values[values > 0]
    if values.size < 100:
        return image.astype(np.float32) / 255.0
    low, high = np.percentile(values, (1.0, 99.5))
    if high <= low:
        return image.astype(np.float32) / 255.0
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def vessel_response(image: np.ndarray, bright: bool) -> np.ndarray:
    unit = robust_unit(image)
    smooth = cv2.GaussianBlur(unit, (0, 0), 4.0)
    response = unit - smooth if bright else smooth - unit
    response = np.clip(response, 0.0, None)
    high = float(np.percentile(response, 99.5))
    return np.clip(response / max(high, 1e-6), 0.0, 1.0)


def make_qa_strip(
    cfp: np.ndarray,
    early: np.ndarray,
    late: np.ndarray,
    warped_early: np.ndarray,
    warped_late: np.ndarray,
    early_homography: np.ndarray,
    late_homography: np.ndarray,
) -> np.ndarray:
    target_height, target_width = cfp.shape[:2]
    source_height, source_width = late.shape[:2]
    corners = np.float32(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]]
    ).reshape(-1, 1, 2)
    cfp_outline = cfp.copy()
    for homography, color in (
        (early_homography, (0, 255, 255)),
        (late_homography, (255, 0, 255)),
    ):
        mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if bool(np.isfinite(mapped).all()):
            cv2.polylines(
                cfp_outline,
                [np.rint(mapped).astype(np.int32)],
                isClosed=True,
                color=color,
                thickness=max(2, target_width // 700),
            )

    target_gray = cv2.cvtColor(cfp, cv2.COLOR_BGR2GRAY)
    cfp_vessel = vessel_response(target_gray, bright=False)

    def overlay_for(phase: np.ndarray) -> np.ndarray:
        ffa_vessel = vessel_response(phase, bright=True)
        overlay = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        overlay[..., 2] = np.rint(cfp_vessel * 255.0).astype(np.uint8)
        overlay[..., 1] = np.rint(
            np.minimum(cfp_vessel, ffa_vessel) * 255.0
        ).astype(np.uint8)
        overlay[..., 0] = np.rint(ffa_vessel * 255.0).astype(np.uint8)
        return overlay

    panel_size = (384, 256)
    panels = [
        cv2.resize(cfp_outline, panel_size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            cv2.cvtColor(warped_early, cv2.COLOR_GRAY2BGR),
            panel_size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(overlay_for(warped_early), panel_size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            cv2.cvtColor(warped_late, cv2.COLOR_GRAY2BGR),
            panel_size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(overlay_for(warped_late), panel_size, interpolation=cv2.INTER_AREA),
    ]
    labels = (
        "CFP + phase FOVs",
        "registered early FFA",
        "early vessel overlay",
        "registered late FFA",
        "late vessel overlay",
    )
    for panel, label in zip(panels, labels):
        cv2.putText(
            panel,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def read_inputs(
    data_root: Path,
    split: str,
    case_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = data_root / split
    cfp = cv2.imread(str(base / "images" / f"{case_id}.png"), cv2.IMREAD_COLOR)
    early = cv2.imread(str(base / "FFA_A" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
    late = cv2.imread(str(base / "FFA_AV" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
    roi = cv2.imread(str(base / "masks" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
    if any(image is None for image in (cfp, early, late, roi)):
        raise FileNotFoundError(f"Missing CFP/FFA/ROI input for {split}/{case_id}")
    if early.shape != late.shape:
        raise ValueError(f"FFA phase shapes differ for {split}/{case_id}")
    return cfp, early, late, roi


def estimate_transform(
    matcher: object,
    source: np.ndarray,
    cfp: np.ndarray,
    roi: np.ndarray,
    context: str,
    ransac_threshold: float,
    minimum_matches: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
    minimum_roi_coverage: float,
) -> tuple[np.ndarray, dict[str, object]]:
    target_height, target_width = cfp.shape[:2]
    source_height, source_width = source.shape[:2]
    match_result = matcher(source, cfp)
    source_points = np.asarray(match_result["mkpts0"], dtype=np.float32)
    target_points = np.asarray(match_result["mkpts1"], dtype=np.float32)
    confidences = np.asarray(match_result["mconf"], dtype=np.float32)
    match_count = int(len(source_points))
    if match_count < minimum_matches:
        raise RuntimeError(f"{context}: only {match_count} MINIMA matches")

    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        ransac_threshold,
        None,
        10_000,
        0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError(f"{context}: homography estimation failed")
    homography = np.asarray(homography, dtype=np.float64)
    if (
        not bool(np.isfinite(homography).all())
        or abs(float(homography[2, 2])) < 1e-12
    ):
        raise RuntimeError(f"{context}: invalid homography")
    homography /= homography[2, 2]
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, match_count)

    projected = cv2.perspectiveTransform(
        source_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    reprojection = np.linalg.norm(projected - target_points, axis=1)
    inlier_reprojection = reprojection[inliers]

    source_support = (source > 0).astype(np.uint8)
    warped_support = cv2.warpPerspective(
        source_support,
        homography,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    target_roi = roi > 127
    roi_coverage = float((warped_support & target_roi).sum()) / max(
        1, int(target_roi.sum())
    )
    support_precision = float((warped_support & target_roi).sum()) / max(
        1, int(warped_support.sum())
    )

    corners = np.float32(
        [
            [0, 0],
            [source_width - 1, 0],
            [source_width - 1, source_height - 1],
            [0, source_height - 1],
        ]
    ).reshape(-1, 1, 2)
    mapped_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    corner_area_ratio = polygon_area(mapped_corners) / max(
        1.0, float(target_width * target_height)
    )
    quality_passed = bool(
        match_count >= minimum_matches
        and inlier_count >= minimum_inliers
        and inlier_ratio >= minimum_inlier_ratio
        and roi_coverage >= minimum_roi_coverage
        and 0.20 <= corner_area_ratio <= 5.0
        and bool(np.isfinite(mapped_corners).all())
    )
    metrics: dict[str, object] = {
        "matches": match_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "confidence_median": (
            float(np.median(confidences)) if confidences.size else 0.0
        ),
        "inlier_reprojection_median_px": (
            float(np.median(inlier_reprojection))
            if inlier_reprojection.size
            else math.inf
        ),
        "inlier_reprojection_p95_px": (
            float(np.percentile(inlier_reprojection, 95))
            if inlier_reprojection.size
            else math.inf
        ),
        "roi_coverage": roi_coverage,
        "support_precision": support_precision,
        "corner_area_ratio": corner_area_ratio,
        "homography": homography.tolist(),
        "match_seconds": float(match_result.get("match_time", 0.0)),
        "quality_passed": quality_passed,
    }
    return homography, metrics


def register_case(
    matcher: object,
    data_root: Path,
    output_root: Path,
    split: str,
    case_id: str,
    ransac_threshold: float,
    minimum_matches: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
    minimum_roi_coverage: float,
) -> dict[str, object]:
    started = time.perf_counter()
    cfp, early, late, roi = read_inputs(data_root, split, case_id)
    target_height, target_width = cfp.shape[:2]
    source_height, source_width = late.shape[:2]
    early_homography, early_metrics = estimate_transform(
        matcher,
        early,
        cfp,
        roi,
        f"{split}/{case_id}/early",
        ransac_threshold,
        minimum_matches,
        minimum_inliers,
        minimum_inlier_ratio,
        minimum_roi_coverage,
    )
    late_homography, late_metrics = estimate_transform(
        matcher,
        late,
        cfp,
        roi,
        f"{split}/{case_id}/late",
        ransac_threshold,
        minimum_matches,
        minimum_inliers,
        minimum_inlier_ratio,
        minimum_roi_coverage,
    )

    warped_early = cv2.warpPerspective(
        early,
        early_homography,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_late = cv2.warpPerspective(
        late,
        late_homography,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    split_output = output_root / split
    (split_output / "FFA_A").mkdir(parents=True, exist_ok=True)
    (split_output / "FFA_AV").mkdir(parents=True, exist_ok=True)
    (split_output / "transforms").mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(split_output / "FFA_A" / f"{case_id}.png"), warped_early):
        raise OSError(f"Failed to write registered early FFA for {split}/{case_id}")
    if not cv2.imwrite(str(split_output / "FFA_AV" / f"{case_id}.png"), warped_late):
        raise OSError(f"Failed to write registered late FFA for {split}/{case_id}")

    result: dict[str, object] = {
        "split": split,
        "id": case_id,
        "source_shape": [source_height, source_width],
        "target_shape": [target_height, target_width],
        "matches": min(
            int(early_metrics["matches"]),
            int(late_metrics["matches"]),
        ),
        "inliers": min(
            int(early_metrics["inliers"]),
            int(late_metrics["inliers"]),
        ),
        "inlier_ratio": min(
            float(early_metrics["inlier_ratio"]),
            float(late_metrics["inlier_ratio"]),
        ),
        "roi_coverage": min(
            float(early_metrics["roi_coverage"]),
            float(late_metrics["roi_coverage"]),
        ),
        "quality_passed": bool(
            early_metrics["quality_passed"] and late_metrics["quality_passed"]
        ),
        "phases": {
            "early": early_metrics,
            "late": late_metrics,
        },
        "total_seconds": time.perf_counter() - started,
    }
    (split_output / "transforms" / f"{case_id}.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    result["_qa_data"] = (
        cfp,
        early,
        late,
        warped_early,
        warped_late,
        early_homography,
        late_homography,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minima-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("training", "validation"),
        default=("training", "validation"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--match-threshold", type=float, default=0.20)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--minimum-matches", type=int, default=8)
    parser.add_argument("--minimum-inliers", type=int, default=8)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-roi-coverage", type=float, default=0.45)
    parser.add_argument("--qa-count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-quality-failures", action="store_true")
    args = parser.parse_args()

    cv2.setRNGSeed(77)
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    minima_root = Path(args.minima_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_MINIMA_LOFTR_SHA256:
        raise RuntimeError(
            "MINIMA-LoFTR checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256}"
        )
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output root is not empty; use a fresh path or --force: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(minima_root))
    from load_model import load_model

    matcher = load_model(
        "loftr",
        SimpleNamespace(ckpt=str(checkpoint), thr=args.match_threshold),
        use_path=False,
    )
    selected_ids = set(args.case_id)
    work: list[tuple[str, str]] = []
    for split in args.splits:
        ids = sorted(path.stem for path in (data_root / split / "images").glob("g_*.png"))
        if selected_ids:
            ids = [case_id for case_id in ids if case_id in selected_ids]
        work.extend((split, case_id) for case_id in ids)
    if not work:
        raise RuntimeError("No CFP/FFA cases selected")

    public_results: list[dict[str, object]] = []
    qa_payloads: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    for split, case_id in work:
        result = register_case(
            matcher=matcher,
            data_root=data_root,
            output_root=output_root,
            split=split,
            case_id=case_id,
            ransac_threshold=args.ransac_threshold,
            minimum_matches=args.minimum_matches,
            minimum_inliers=args.minimum_inliers,
            minimum_inlier_ratio=args.minimum_inlier_ratio,
            minimum_roi_coverage=args.minimum_roi_coverage,
        )
        qa_payloads[(split, case_id)] = result.pop("_qa_data")
        public_results.append(result)
        print(json.dumps(result), flush=True)

    quality_failures = [
        result for result in public_results if not bool(result["quality_passed"])
    ]
    ranked = sorted(
        public_results,
        key=lambda result: (
            bool(result["quality_passed"]),
            float(result["roi_coverage"]),
            float(result["inlier_ratio"]),
        ),
    )
    qa_dir = output_root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for result in ranked[: max(0, args.qa_count)]:
        split = str(result["split"])
        case_id = str(result["id"])
        strip = make_qa_strip(*qa_payloads[(split, case_id)])
        cv2.imwrite(str(qa_dir / f"{split}_{case_id}.jpg"), strip)

    summary = {
        "method": "MINIMA-LoFTR",
        "source": "https://github.com/LSXI7/MINIMA",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "labels_used": False,
        "transform_direction": "each FFA phase -> CFP",
        "same_transform_for_early_and_late": False,
        "configuration": {
            "match_threshold": args.match_threshold,
            "ransac": "cv2.RANSAC homography",
            "ransac_threshold_px": args.ransac_threshold,
            "minimum_matches": args.minimum_matches,
            "minimum_inliers": args.minimum_inliers,
            "minimum_inlier_ratio": args.minimum_inlier_ratio,
            "minimum_roi_coverage": args.minimum_roi_coverage,
        },
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "cases": len(public_results),
        "quality_passed": len(public_results) - len(quality_failures),
        "quality_failed": len(quality_failures),
        "minimum_matches_observed": min(int(result["matches"]) for result in public_results),
        "minimum_inliers_observed": min(int(result["inliers"]) for result in public_results),
        "minimum_inlier_ratio_observed": min(
            float(result["inlier_ratio"]) for result in public_results
        ),
        "minimum_roi_coverage_observed": min(
            float(result["roi_coverage"]) for result in public_results
        ),
        "median_roi_coverage": float(
            np.median([float(result["roi_coverage"]) for result in public_results])
        ),
        "results": public_results,
    }
    (output_root / "registration_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if quality_failures and not args.allow_quality_failures:
        failed = ", ".join(
            f"{result['split']}/{result['id']}" for result in quality_failures
        )
        raise RuntimeError(f"Registration quality gate failed: {failed}")


if __name__ == "__main__":
    main()
