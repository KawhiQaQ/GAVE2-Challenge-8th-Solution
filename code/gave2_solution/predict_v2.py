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
from gave2v1.model_v2 import build_v2


def _flip(
    tensor: torch.Tensor | None,
    horizontal: bool,
    vertical: bool,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    dimensions: list[int] = []
    if vertical:
        dimensions.append(-2)
    if horizontal:
        dimensions.append(-1)
    return torch.flip(tensor, dimensions) if dimensions else tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write GAVE2 V2 RGB probabilities in [A, vessel, V] order."
    )
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--ffa-root",
        help=(
            "Optional root containing split/FFA_A and split/FFA_AV. "
            "Use the same registered cache recorded by the checkpoint recipe."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="validation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-validation-only", action="store_true")
    parser.add_argument(
        "--tta-flips",
        action="store_true",
        help=(
            "Average identity, horizontal, vertical, and horizontal+vertical "
            "predictions after mapping them back to the native orientation."
        ),
    )
    parser.add_argument(
        "--fuse-centerlines",
        action="store_true",
        help=(
            "Raise A/V probabilities with their trained centerline heads. "
            "The vessel channel is unchanged; the default preserves V2."
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
    if int(config["task"]) != args.task:
        raise ValueError("Checkpoint task does not match --task")
    size = (int(config["height"]), int(config["width"]))
    model = build_v2(
        args.task,
        pretrained=False,
        num_refinements=int(config["num_refinements"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = select_device(args.device)
    model.to(device).eval()

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve() if args.ffa_root else None
    ids = (
        list(checkpoint["split"]["validation"])
        if args.checkpoint_validation_only
        else case_ids(data_root, args.split)
    )
    dataset = GAVE2Dataset(
        data_root,
        ids,
        task=args.task,
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
            ffa = sample.get("ffa")
            if ffa is not None:
                ffa = ffa.unsqueeze(0).to(device)
            transforms = (
                ((False, False), (False, True), (True, False), (True, True))
                if args.tta_flips
                else ((False, False),)
            )
            probability_sum: torch.Tensor | None = None
            for horizontal, vertical in transforms:
                transformed_rgb = _flip(rgb, horizontal, vertical)
                transformed_ffa = _flip(ffa, horizontal, vertical)
                if transformed_rgb is None:
                    raise RuntimeError("RGB tensor unexpectedly missing")
                outputs = model(transformed_rgb, transformed_ffa)
                transformed_probability = model.challenge_probabilities(outputs)
                if args.fuse_centerlines:
                    centerline_logits = outputs["centerline_logits"]
                    if not isinstance(centerline_logits, torch.Tensor):
                        raise TypeError(
                            "outputs['centerline_logits'] must be a tensor"
                        )
                    centerline_probability = torch.sigmoid(
                        centerline_logits[:, (0, 2)]
                    )
                    transformed_probability[:, 0] = torch.maximum(
                        transformed_probability[:, 0],
                        centerline_probability[:, 0],
                    )
                    transformed_probability[:, 2] = torch.maximum(
                        transformed_probability[:, 2],
                        centerline_probability[:, 1],
                    )
                probability = _flip(
                    transformed_probability,
                    horizontal,
                    vertical,
                )
                if probability is None:
                    raise RuntimeError("Probability tensor unexpectedly missing")
                probability_sum = (
                    probability
                    if probability_sum is None
                    else probability_sum + probability
                )
            if probability_sum is None:
                raise RuntimeError("No inference transforms were evaluated")
            probabilities = probability_sum / float(len(transforms))
            probabilities = probabilities * sample["roi"].to(device).unsqueeze(0)
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
            probabilities_array = (
                probabilities[0].clamp(0.0, 1.0).cpu().numpy()
            )
            probabilities_array *= roi[None]
            image = np.rint(
                probabilities_array.transpose(1, 2, 0) * 255.0
            ).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(
                output_dir / f"{case_id}.png"
            )
            print(json.dumps({"id": case_id}), flush=True)

    print(
        json.dumps(
            {
                "task": args.task,
                "checkpoint": str(checkpoint_path),
                "written": len(ids),
                "output_dir": str(output_dir),
                "fuse_centerlines": args.fuse_centerlines,
                "tta_flips": args.tta_flips,
                "ffa_root": (
                    str(ffa_root) if ffa_root is not None else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
