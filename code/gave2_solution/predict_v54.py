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
from gave2v1.model_v54 import build_v54


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write V54 probabilities in [A, vessel, V]."
    )
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ffa-root")
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="training",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-validation-only", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if int(config["task"]) != args.task:
        raise ValueError("checkpoint task does not match --task")
    model = build_v54(
        task=args.task,
        pretrained=False,
        r2_steps=int(config["r2_steps"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = select_device(args.device)
    model.to(device).eval()

    data_root = Path(args.data_root).resolve()
    ffa_root = Path(args.ffa_root).resolve() if args.ffa_root else None
    if args.task == 2 and ffa_root is None:
        raise ValueError("Task2 requires --ffa-root")
    if args.task == 1 and ffa_root is not None:
        raise ValueError("Task1 must not receive --ffa-root")
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
        size=(int(config["height"]), int(config["width"])),
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
            outputs = model(rgb, ffa)
            probabilities = (
                model.challenge_probabilities(outputs)
                * sample["roi"].to(device).unsqueeze(0)
            )
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
                "task": args.task,
                "checkpoint": str(checkpoint_path),
                "written": len(ids),
                "output_dir": str(output_dir),
                "public_r2_weights_loaded": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

