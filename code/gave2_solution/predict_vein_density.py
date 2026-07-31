#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from gave2v1.data import GAVE2Dataset, case_ids
from gave2v1.engine import select_device
from gave2v1.model_vein_refinement import (
    VeinDensityRefinementNet,
    load_compact_refiner_state,
    load_vein_parent,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write VascFusion-Quant C-zone vein-density probability maps."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root", required=True)
    parser.add_argument("--zone-c-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="validation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-validation-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    parent, parent_report = load_vein_parent(args.parent_checkpoint)
    model = VeinDensityRefinementNet(parent)
    load_report = load_compact_refiner_state(
        model,
        checkpoint["refiner"],
    )
    device = select_device(args.device)
    model.to(device).eval()

    data_root = Path(args.data_root).resolve()
    ids = (
        list(checkpoint["split"]["validation"])
        if args.checkpoint_validation_only
        else case_ids(data_root, args.split)
    )
    dataset = GAVE2Dataset(
        data_root,
        ids,
        task=2,
        split=args.split,
        size=(int(config["height"]), int(config["width"])),
        augment=False,
        cache=False,
        ffa_root=args.ffa_root,
        region_dir=args.zone_c_dir,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for sample in dataset:
            rgb = sample["rgb"].unsqueeze(0).to(device)
            ffa = sample["ffa"].unsqueeze(0).to(device)
            region = sample["region"].unsqueeze(0).to(device)
            probabilities = model(
                rgb,
                ffa,
                region,
            )["challenge_probabilities"]
            probabilities = probabilities * sample["roi"].to(
                device
            ).unsqueeze(0)
            with Image.open(
                data_root
                / args.split
                / "images"
                / f"{sample['id']}.png"
            ) as original:
                original_width, original_height = original.size
            if probabilities.shape[-2:] != (
                original_height,
                original_width,
            ):
                probabilities = F.interpolate(
                    probabilities,
                    size=(original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                )
            with Image.open(
                data_root
                / args.split
                / "masks"
                / f"{sample['id']}.png"
            ) as roi_image:
                roi = (
                    np.asarray(roi_image.convert("L"), dtype=np.uint8)
                    > 127
                )
            array = probabilities[0].clamp(0.0, 1.0).cpu().numpy()
            array *= roi[None]
            image = np.rint(array.transpose(1, 2, 0) * 255.0).astype(
                np.uint8
            )
            Image.fromarray(image, mode="RGB").save(
                output_dir / f"{sample['id']}.png"
            )
            print(json.dumps({"id": sample["id"]}), flush=True)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "parent": parent_report,
                "compact_load": load_report,
                "written": len(ids),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
