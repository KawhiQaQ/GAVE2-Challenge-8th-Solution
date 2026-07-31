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
from gave2v1.model_phase_geometry import build_phase_geometry_net


def _flip(tensor: torch.Tensor, horizontal: bool, vertical: bool) -> torch.Tensor:
    dimensions: list[int] = []
    if vertical:
        dimensions.append(-2)
    if horizontal:
        dimensions.append(-1)
    return torch.flip(tensor, dimensions) if dimensions else tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write VascFusion-Quant phase-geometry probabilities in "
            "[A, vessel, V]."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root", required=True)
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="training",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-validation-only", action="store_true")
    parser.add_argument(
        "--tta-flips",
        action="store_true",
        help=(
            "Average identity, horizontal, vertical, and horizontal+vertical "
            "predictions after mapping them back to the native orientation."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    size = (int(config["height"]), int(config["width"]))
    model = build_phase_geometry_net(
        phase_pretrained=False,
        num_refinements=int(config["num_refinements"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = select_device(args.device)
    model.to(device).eval()

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve()
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
        size=size,
        augment=False,
        cache=False,
        ffa_root=ffa_root,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for sample in dataset:
            case_id = sample["id"]
            rgb = sample["rgb"].unsqueeze(0).to(device)
            ffa = sample["ffa"].unsqueeze(0).to(device)
            transforms = (
                ((False, False), (False, True), (True, False), (True, True))
                if args.tta_flips
                else ((False, False),)
            )
            probability_sum: torch.Tensor | None = None
            for horizontal, vertical in transforms:
                outputs = model(
                    _flip(rgb, horizontal, vertical),
                    _flip(ffa, horizontal, vertical),
                )
                transformed_probability = model.challenge_probabilities(outputs)
                probability = _flip(
                    transformed_probability,
                    horizontal,
                    vertical,
                )
                probability_sum = (
                    probability
                    if probability_sum is None
                    else probability_sum + probability
                )
            if probability_sum is None:
                raise RuntimeError("No inference transforms were evaluated")
            probabilities = (
                probability_sum / float(len(transforms))
            ) * sample["roi"].to(device).unsqueeze(0)
            with Image.open(
                data_root / args.split / "images" / f"{case_id}.png"
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
                data_root / args.split / "masks" / f"{case_id}.png"
            ) as roi_image:
                roi = np.asarray(
                    roi_image.convert("L"),
                    dtype=np.uint8,
                ) > 127
            array = probabilities[0].clamp(0.0, 1.0).cpu().numpy()
            array *= roi[None]
            image = np.rint(array.transpose(1, 2, 0) * 255.0).astype(
                np.uint8
            )
            Image.fromarray(image, mode="RGB").save(
                output_dir / f"{case_id}.png"
            )
            print(json.dumps({"id": case_id}), flush=True)

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "written": len(ids),
                "output_dir": str(output_dir),
                "tta_flips": args.tta_flips,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
