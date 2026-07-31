#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import dilation, disk, skeletonize


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-4, 1.0 - 1e-4)
    return np.log(probability) - np.log1p(-probability)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -12.0, 12.0)))


def branch_consistent_probabilities(
    probability: np.ndarray,
    roi: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    artery = probability[..., 0].astype(np.float32) / 255.0
    vessel = probability[..., 1].astype(np.float32) / 255.0
    vein = probability[..., 2].astype(np.float32) / 255.0
    vessel_support = (vessel > 0.5) & roi
    skeleton = skeletonize(vessel_support)

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    degree = ndi.convolve(
        skeleton.astype(np.uint8),
        neighbor_kernel,
        mode="constant",
        cval=0,
    )
    junction = skeleton & (degree >= 3)
    junction_zone = dilation(junction, footprint=disk(3))
    branch_skeleton = skeleton & ~dilation(
        junction,
        footprint=disk(1),
    )
    branches = label(branch_skeleton, connectivity=2)

    original_score = logit(artery) - logit(vein)
    branch_scores = np.zeros(int(branches.max()) + 1, dtype=np.float32)
    valid_branch = np.zeros_like(branch_scores, dtype=bool)
    valid_branch_pixels = 0
    for branch_id in range(1, len(branch_scores)):
        coordinates = branches == branch_id
        length = int(coordinates.sum())
        if length < 6:
            continue
        branch_scores[branch_id] = float(
            np.median(original_score[coordinates])
        )
        valid_branch[branch_id] = True
        valid_branch_pixels += length

    labeled_skeleton = valid_branch[branches]
    if not bool(labeled_skeleton.any()):
        return probability.copy(), {
            "branches": 0,
            "valid_branch_skeleton_pixels": 0,
            "changed_pixels": 0,
            "overlap_before": int(
                ((artery > 0.5) & (vein > 0.5) & roi).sum()
            ),
            "overlap_after": int(
                ((artery > 0.5) & (vein > 0.5) & roi).sum()
            ),
        }

    nearest = ndi.distance_transform_edt(
        ~labeled_skeleton,
        return_distances=False,
        return_indices=True,
    )
    nearest_branch = branches[tuple(nearest)]
    applicable = (
        vessel_support
        & valid_branch[nearest_branch]
        & ~junction_zone
    )
    propagated_score = branch_scores[nearest_branch]
    consistent_score = (
        0.25 * original_score + 0.75 * propagated_score
    )
    artery_fraction = sigmoid(consistent_score)
    class_mass = artery + vein
    consistent_artery = np.clip(
        class_mass * artery_fraction,
        0.0,
        1.0,
    )
    consistent_vein = np.clip(
        class_mass * (1.0 - artery_fraction),
        0.0,
        1.0,
    )
    output_artery = np.where(
        applicable,
        consistent_artery,
        artery,
    )
    output_vein = np.where(
        applicable,
        consistent_vein,
        vein,
    )
    output = np.stack(
        (output_artery, vessel, output_vein),
        axis=-1,
    )
    output *= roi[..., None]
    output_uint8 = np.rint(output * 255.0).astype(np.uint8)
    changed = (
        (output_uint8[..., 0] != probability[..., 0])
        | (output_uint8[..., 2] != probability[..., 2])
    )
    return output_uint8, {
        "branches": int(valid_branch.sum()),
        "valid_branch_skeleton_pixels": valid_branch_pixels,
        "changed_pixels": int(changed.sum()),
        "overlap_before": int(
            ((artery > 0.5) & (vein > 0.5) & roi).sum()
        ),
        "overlap_after": int(
            (
                (output_artery > 0.5)
                & (output_vein > 0.5)
                & roi
            ).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce robust A/V identity within predicted vessel branches "
            "while preserving crossing neighborhoods."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.glob("g_*.png")):
        with Image.open(path) as image:
            probability = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            ).copy()
        with Image.open(masks_dir / path.name) as image:
            roi = np.asarray(
                image.convert("L"),
                dtype=np.uint8,
            ) > 127
        output, metrics = branch_consistent_probabilities(
            probability,
            roi,
        )
        Image.fromarray(output, mode="RGB").save(output_dir / path.name)
        row = {"id": path.stem, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(
        json.dumps(
            {
                "written": len(rows),
                "output_dir": str(output_dir.resolve()),
                "total_changed_pixels": sum(
                    int(row["changed_pixels"]) for row in rows
                ),
                "overlap_before": sum(
                    int(row["overlap_before"]) for row in rows
                ),
                "overlap_after": sum(
                    int(row["overlap_after"]) for row in rows
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
