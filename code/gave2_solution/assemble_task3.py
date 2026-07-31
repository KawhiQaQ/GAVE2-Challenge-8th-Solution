#!/usr/bin/env python3
"""Assemble a frozen GAVE2 Task3 policy from validated component sources.

The default V3 policy uses:

* CRAE, CRVE and AVR from independently thresholded artery/vein masks;
* densities and fractal dimensions from vessel-gated A/V masks;
* one global out-of-fold multiplicative calibration per density.

No per-case choices or validation-label-dependent rules are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BIOMARKER_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)
DENSITY_KEYS = ("artery_density", "vein_density")
CALIBER_KEYS = ("CRAE", "CRVE", "AVR")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_biomarkers(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"{path}: malformed line {line!r}")
        key = fields[0]
        if key in values:
            raise ValueError(f"{path}: duplicate key {key}")
        values[key] = float(fields[1])
    if tuple(values) != BIOMARKER_KEYS:
        raise ValueError(f"{path}: unexpected biomarker keys/order")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{path}: non-finite value")
    return values


def expected_ids(start: int, count: int) -> list[str]:
    return [f"g_{index:03d}" for index in range(start, start + count)]


def validate_ids(path: Path, case_ids: list[str]) -> None:
    actual = sorted(item.stem for item in path.glob("g_*.txt"))
    if actual != case_ids:
        raise ValueError(
            f"{path}: expected {len(case_ids)} case files, found {len(actual)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Task3 TXT directory generated with vessel_argmax masks.",
    )
    parser.add_argument(
        "--caliber-dir",
        required=True,
        help="Task3 TXT directory generated with independent_av masks.",
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-output")
    parser.add_argument("--start-index", type=int, default=51)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    caliber_dir = Path(args.caliber_dir).resolve()
    calibration_path = Path(args.calibration).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    case_ids = expected_ids(args.start_index, args.count)
    validate_ids(base_dir, case_ids)
    validate_ids(caliber_dir, case_ids)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    factors = {
        key: float(calibration["factors"][key])
        for key in DENSITY_KEYS
    }
    if not all(math.isfinite(value) and value > 0 for value in factors.values()):
        raise ValueError("Density calibration factors must be finite and positive")

    output_dir.mkdir(parents=True)
    changed: dict[str, object] = {}
    for case_id in case_ids:
        base_path = base_dir / f"{case_id}.txt"
        caliber_path = caliber_dir / f"{case_id}.txt"
        base = parse_biomarkers(base_path)
        caliber = parse_biomarkers(caliber_path)

        if caliber["CRVE"] > 0:
            ratio = caliber["CRAE"] / caliber["CRVE"]
            if abs(ratio - caliber["AVR"]) > 2e-4:
                raise ValueError(f"{caliber_path}: inconsistent AVR")

        output = dict(base)
        for key in CALIBER_KEYS:
            output[key] = caliber[key]
        for key in DENSITY_KEYS:
            output[key] = base[key] * factors[key]

        destination = output_dir / f"{case_id}.txt"
        destination.write_text(
            "".join(f"{key} {output[key]:.6f}\n" for key in BIOMARKER_KEYS),
            encoding="utf-8",
        )
        changed[case_id] = {
            "caliber": {
                key: {"base": base[key], "assembled": output[key]}
                for key in CALIBER_KEYS
            },
            "density": {
                key: {
                    "base": base[key],
                    "factor": factors[key],
                    "assembled": output[key],
                }
                for key in DENSITY_KEYS
            },
            "fractal_dimensions_unchanged": all(
                output[key] == base[key]
                for key in (
                    "artery_fractal_dimension",
                    "vein_fractal_dimension",
                )
            ),
        }

    report = {
        "schema_version": 1,
        "policy": {
            "caliber_source": "independent_av",
            "density_source": "vessel_argmax_then_global_oof_scale",
            "fractal_dimension_source": "vessel_argmax",
            "per_case_selection": False,
        },
        "base_dir": str(base_dir),
        "caliber_dir": str(caliber_dir),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "factors": factors,
        "case_count": len(case_ids),
        "changed": changed,
    }
    report_output = (
        Path(args.report_output).resolve()
        if args.report_output
        else output_dir / "assembly_report.json"
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "case_count": len(case_ids),
                "factors": factors,
                "report": str(report_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
