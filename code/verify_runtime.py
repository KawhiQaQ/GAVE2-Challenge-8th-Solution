#!/usr/bin/env python3
"""Validate the VascFusion runtime, mounted data, and released checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from PIL import Image
import scipy
import sklearn
import skimage
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "weights_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_weights(
    weights_dir: Path,
    require_inference: bool,
    require_initialization: bool,
) -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    missing: list[str] = []
    required_missing: list[str] = []
    mismatched: list[str] = []
    for item in payload["files"]:
        relative = Path(item["path"]).relative_to("weights")
        path = weights_dir / relative
        if not path.is_file():
            missing.append(str(relative))
            is_initialization = relative.parts[0] == "initialization"
            if (is_initialization and require_initialization) or (
                not is_initialization and require_inference
            ):
                required_missing.append(str(relative))
            continue
        actual = sha256(path)
        ok = actual == item["sha256"]
        results.append({"path": str(relative), "sha256_ok": ok})
        if not ok:
            mismatched.append(str(relative))
    if mismatched:
        raise RuntimeError("Checkpoint SHA256 mismatch: " + ", ".join(mismatched))
    if required_missing:
        raise FileNotFoundError(
            "Missing required checkpoints: " + ", ".join(required_missing)
        )
    return {"checked": results, "missing": missing}


def validate_data(data_root: Path) -> dict[str, object]:
    required = ["images", "masks", "FFA_A", "FFA_AV"]
    split = data_root / "validation"
    missing = [name for name in required if not (split / name).is_dir()]
    if missing:
        raise FileNotFoundError("Missing validation directories: " + ", ".join(missing))
    counts = {name: len(list((split / name).glob("g_*.png"))) for name in required}
    if not counts["images"] or len(set(counts.values())) != 1:
        raise RuntimeError(f"Inconsistent validation case counts: {counts}")
    return {"root": str(data_root), "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", default=str(ROOT / "weights"))
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--require-initialization", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--data-root")
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Start Docker with --gpus all and the "
            "NVIDIA Container Toolkit installed on the host."
        )

    report: dict[str, object] = {
        "status": "ok",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "scikit_learn": sklearn.__version__,
        "h5py": h5py.__version__,
        "weights": validate_weights(
            Path(args.weights_dir),
            args.require_weights,
            args.require_initialization,
        ),
    }
    if args.data_root:
        report["data"] = validate_data(Path(args.data_root))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
