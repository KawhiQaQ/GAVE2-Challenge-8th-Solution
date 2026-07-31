#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import dilation, disk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply fixed-resolution structural A/V probability growth."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument(
        "--vessel-gate",
        type=float,
        default=None,
        help=(
            "If set, only newly grown A/V pixels supported by at least this "
            "green-channel vessel probability are retained."
        ),
    )
    args = parser.parse_args()

    if args.radius < 0:
        raise ValueError("--radius must be non-negative")
    if args.vessel_gate is not None and not 0.0 <= args.vessel_gate <= 1.0:
        raise ValueError("--vessel-gate must be in [0, 1]")
    input_dir = Path(args.input_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    footprint = disk(args.radius)

    written = 0
    for path in sorted(input_dir.glob("g_*.png")):
        with Image.open(path) as image:
            probability = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(masks_dir / path.name) as image:
            roi = np.asarray(image.convert("L"), dtype=np.uint8) > 127
        vessel_support = (
            probability[..., 1]
            >= round(255.0 * args.vessel_gate)
            if args.vessel_gate is not None
            else None
        )
        for channel in (0, 2):
            original = probability[..., channel].copy()
            grown = dilation(
                original,
                footprint=footprint,
            )
            if vessel_support is not None:
                grown = np.where(vessel_support, grown, 0).astype(np.uint8)
            probability[..., channel] = np.maximum(original, grown)
        probability *= roi[..., None]
        Image.fromarray(probability, mode="RGB").save(output_dir / path.name)
        written += 1
    print(
        f"written={written} radius={args.radius} "
        f"vessel_gate={args.vessel_gate} output={output_dir}"
    )


if __name__ == "__main__":
    main()
