#!/usr/bin/env python3
"""Assemble the seven VascFusion-Quant biomarker fields.

The measurement strategy is fixed for the complete dataset:

* CRAE, CRVE, AVR: caliber-oriented probability source;
* artery_density: globally calibrated artery-density source;
* vein_density: compact C-zone vein-density refinement source;
* artery_fractal_dimension: branch-consistent artery geometry;
* vein_fractal_dimension: arithmetic mean of phase-geometry estimates at
  thresholds 0.4, 0.5 and 0.6.

No labels, per-case gates, or hidden leaderboard information are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


KEYS = (
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


def parse_task3(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces = line.split()
        if len(pieces) < 2:
            raise ValueError(f"{path}: malformed line {line!r}")
        key = pieces[0]
        if key in values:
            raise ValueError(f"{path}: duplicate key {key}")
        values[key] = float(pieces[1])
    if tuple(values) != KEYS:
        raise ValueError(f"{path}: unexpected keys or order")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{path}: non-finite value")
    return values


def task3_ids(path: Path) -> list[str]:
    return sorted(item.stem for item in path.glob("g_*.txt"))


def load_per_case(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: per_case payload must be an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caliber-dir", type=Path, required=True)
    parser.add_argument("--artery-density-dir", type=Path, required=True)
    parser.add_argument("--vein-density-dir", type=Path, required=True)
    parser.add_argument("--artery-fd-dir", type=Path, required=True)
    parser.add_argument(
        "--vein-fd-per-case",
        type=Path,
        nargs=3,
        required=True,
        metavar=("THRESHOLD_04", "THRESHOLD_05", "THRESHOLD_06"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "caliber": args.caliber_dir.resolve(),
        "artery_density": args.artery_density_dir.resolve(),
        "vein_density": args.vein_density_dir.resolve(),
        "artery_fd": args.artery_fd_dir.resolve(),
    }
    ids = task3_ids(sources["caliber"])
    if not ids:
        raise RuntimeError("Caliber source contains no g_*.txt files")
    for name, path in sources.items():
        if task3_ids(path) != ids:
            raise ValueError(f"{name} cases do not match caliber cases")

    persistence_paths = [path.resolve() for path in args.vein_fd_per_case]
    persistence = [load_per_case(path) for path in persistence_paths]
    for path, payload in zip(persistence_paths, persistence):
        if sorted(payload) != ids:
            raise ValueError(f"{path}: cases do not match Task3 sources")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    cases: dict[str, Any] = {}
    for case_id in ids:
        caliber = parse_task3(sources["caliber"] / f"{case_id}.txt")
        artery_density = parse_task3(
            sources["artery_density"] / f"{case_id}.txt"
        )
        vein_density = parse_task3(
            sources["vein_density"] / f"{case_id}.txt"
        )
        artery_fd = parse_task3(sources["artery_fd"] / f"{case_id}.txt")
        vein_fd_values = [
            float(payload[case_id]["prediction"]["vein_fractal_dimension"])
            for payload in persistence
        ]
        if not all(math.isfinite(value) for value in vein_fd_values):
            raise ValueError(f"{case_id}: non-finite vein FD")

        output = {
            "CRAE": caliber["CRAE"],
            "CRVE": caliber["CRVE"],
            "AVR": caliber["AVR"],
            "artery_density": artery_density["artery_density"],
            "vein_density": vein_density["vein_density"],
            "artery_fractal_dimension": artery_fd[
                "artery_fractal_dimension"
            ],
            "vein_fractal_dimension": sum(vein_fd_values) / 3.0,
        }
        if not all(math.isfinite(value) for value in output.values()):
            raise ValueError(f"{case_id}: assembled non-finite value")
        destination = output_dir / f"{case_id}.txt"
        destination.write_text(
            "".join(f"{key} {output[key]:.6f}\n" for key in KEYS),
            encoding="utf-8",
        )
        cases[case_id] = {
            "sha256": sha256(destination),
            "vein_fd_threshold_values": vein_fd_values,
        }

    report = {
        "schema_version": 1,
        "method": "VascFusion-Quant",
        "case_count": len(ids),
        "case_ids": ids,
        "measurement_route": {
            "CRAE": "caliber expert with original FFA geometry",
            "CRVE": "caliber expert with original FFA geometry",
            "AVR": "caliber expert with original FFA geometry",
            "artery_density": "globally calibrated artery-density expert",
            "vein_density": "registered-FFA C-zone vein refiner",
            "artery_fractal_dimension": (
                "deterministic branch consistency over caliber probabilities"
            ),
            "vein_fractal_dimension": (
                "phase-geometry scalar mean at thresholds 0.4/0.5/0.6"
            ),
        },
        "source_directories": {key: str(value) for key, value in sources.items()},
        "vein_fd_per_case_reports": [
            str(path) for path in persistence_paths
        ],
        "per_case_selection": False,
        "labels_used": False,
        "cases": cases,
    }
    report_output = args.report_output.resolve()
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
