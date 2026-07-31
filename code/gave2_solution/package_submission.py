#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import zipfile

from PIL import Image


BIOMARKER_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ids(start: int, count: int) -> list[str]:
    return [f"g_{index:03d}" for index in range(start, start + count)]


def validate_probability_folder(path: Path, ids: list[str]) -> None:
    actual = sorted(item.stem for item in path.glob("g_*.png"))
    if actual != ids:
        raise ValueError(f"{path}: unexpected PNG names")
    for case_id in ids:
        with Image.open(path / f"{case_id}.png") as image:
            if image.mode != "RGB":
                raise ValueError(f"{case_id}: expected RGB, got {image.mode}")
            if image.size != (1536, 1024):
                raise ValueError(f"{case_id}: expected 1536x1024, got {image.size}")


def validate_biomarker_folder(path: Path, ids: list[str]) -> None:
    actual = sorted(item.stem for item in path.glob("g_*.txt"))
    if actual != ids:
        raise ValueError(f"{path}: unexpected TXT names")
    for case_id in ids:
        values: dict[str, float] = {}
        for line in (path / f"{case_id}.txt").read_text(encoding="utf-8").splitlines():
            key, raw = line.split()[:2]
            values[key] = float(raw)
        if tuple(values) != BIOMARKER_KEYS:
            raise ValueError(f"{case_id}: biomarker keys/order do not match")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{case_id}: non-finite biomarker value")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and zip a GAVE2 submission.")
    parser.add_argument("--submission-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=51)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-size-bytes", type=int, default=100_000_000)
    args = parser.parse_args()

    root = Path(args.submission_root).resolve()
    ids = expected_ids(args.start_index, args.count)
    validate_probability_folder(root / "Task1", ids)
    validate_probability_folder(root / "Task2", ids)
    validate_biomarker_folder(root / "Task3", ids)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt")):
            for case_id in ids:
                source = root / task / f"{case_id}{suffix}"
                archive.write(source, arcname=f"{task}/{source.name}")
    with zipfile.ZipFile(output, "r") as archive:
        bad_entry = archive.testzip()
        names = archive.namelist()
    if bad_entry is not None:
        raise RuntimeError(f"CRC failure in {bad_entry}")
    if len(names) != len(ids) * 3 or len(names) != len(set(names)):
        raise RuntimeError("ZIP entry count or uniqueness check failed")
    size_bytes = output.stat().st_size
    if size_bytes > args.max_size_bytes:
        raise RuntimeError(
            f"ZIP is {size_bytes} bytes, above limit {args.max_size_bytes}"
        )
    print(
        f"validated_and_written {output} cases={len(ids)} tasks=3 "
        f"entries={len(names)} crc_ok=true size_bytes={size_bytes} "
        f"sha256={sha256(output)}"
    )


if __name__ == "__main__":
    main()
