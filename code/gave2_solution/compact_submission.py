#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def threshold_preserving_quantize(
    image: np.ndarray,
    step: int,
) -> np.ndarray:
    values = image.astype(np.int16)
    lower = values < 128
    quantized = np.empty_like(values)
    quantized[lower] = np.clip(
        ((values[lower] + step // 2) // step) * step,
        0,
        127,
    )
    upper_values = values[~lower] - 128
    quantized[~lower] = 128 + np.clip(
        ((upper_values + step // 2) // step) * step,
        0,
        127,
    )
    result = quantized.astype(np.uint8)
    if not np.array_equal(image >= 128, result >= 128):
        raise RuntimeError("quantization changed a threshold decision")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compact GAVE2 RGB probabilities without changing any 0.5 "
            "threshold decision."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()
    if args.step < 1 or args.step > 32:
        raise ValueError("--step must be in [1, 32]")

    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for task in ("Task1", "Task2", "Task3"):
        (output / task).mkdir(parents=True)

    maximum_error = 0
    png_count = 0
    for task in ("Task1", "Task2"):
        for path in sorted((source / task).glob("g_*.png")):
            with Image.open(path) as image:
                probability = np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                )
            compact = threshold_preserving_quantize(
                probability,
                args.step,
            )
            maximum_error = max(
                maximum_error,
                int(
                    np.abs(
                        compact.astype(np.int16)
                        - probability.astype(np.int16)
                    ).max()
                ),
            )
            Image.fromarray(compact, mode="RGB").save(
                output / task / path.name,
                format="PNG",
                optimize=True,
                compress_level=9,
            )
            png_count += 1

    txt_count = 0
    for path in sorted((source / "Task3").glob("g_*.txt")):
        shutil.copy2(path, output / "Task3" / path.name)
        txt_count += 1
    print(
        f"written_png={png_count} written_txt={txt_count} "
        f"step={args.step} maximum_uint8_error={maximum_error}"
    )


if __name__ == "__main__":
    main()
