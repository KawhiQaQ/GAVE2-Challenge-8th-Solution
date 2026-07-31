#!/usr/bin/env python3
"""Audit one fixed multi-threshold estimator for fractal dimension.

The estimator is deliberately fixed before looking at either validation fold:
calculate the official hard-mask FD at thresholds 0.4, 0.5 and 0.6, then
average the three scalar FD values.  It is a scalar uncertainty marginal,
not a threshold search and not a probability/checkpoint ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = (np.abs(truth) + np.abs(prediction)) / 2.0
    return float(
        100.0
        * np.divide(
            np.abs(truth - prediction),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        ).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-case",
        nargs=3,
        required=True,
        metavar=("THRESHOLD_04", "THRESHOLD_05", "THRESHOLD_06"),
    )
    parser.add_argument("--parent-summary", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sources = [_read(path) for path in args.per_case]
    case_ids = sorted(sources[0])
    if not case_ids or any(sorted(source) != case_ids for source in sources):
        raise RuntimeError("All threshold sources must contain identical cases")
    parent = _read(args.parent_summary)
    fields: dict[str, Any] = {}
    per_case: dict[str, Any] = {}
    for case_id in case_ids:
        truth0 = sources[0][case_id]["truth"]
        for source in sources[1:]:
            for field in FIELDS:
                if not np.isclose(
                    source[case_id]["truth"][field],
                    truth0[field],
                ):
                    raise RuntimeError(
                        f"Truth mismatch for {case_id}/{field}"
                    )
        per_case[case_id] = {
            "truth": {field: float(truth0[field]) for field in FIELDS},
            "prediction": {
                field: float(
                    np.mean(
                        [
                            source[case_id]["prediction"][field]
                            for source in sources
                        ]
                    )
                )
                for field in FIELDS
            },
        }
    for field in FIELDS:
        truth = np.asarray(
            [per_case[case_id]["truth"][field] for case_id in case_ids],
            dtype=np.float64,
        )
        prediction = np.asarray(
            [
                per_case[case_id]["prediction"][field]
                for case_id in case_ids
            ],
            dtype=np.float64,
        )
        mae = float(np.abs(truth - prediction).mean())
        smape = _smape(truth, prediction)
        parent_metrics = parent["metrics"][field]
        ratios = {
            "mae": mae / float(parent_metrics["mae"]),
            "smape_percent": (
                smape / float(parent_metrics["smape_percent"])
            ),
        }
        fields[field] = {
            "candidate": {
                "mae": mae,
                "smape_percent": smape,
                "truth_mean": float(truth.mean()),
                "prediction_mean": float(prediction.mean()),
            },
            "parent": {
                "mae": float(parent_metrics["mae"]),
                "smape_percent": float(
                    parent_metrics["smape_percent"]
                ),
            },
            "error_ratios": ratios,
            "passed": all(value < 1.0 for value in ratios.values()),
        }
    report = {
        "schema_version": 1,
        "candidate": "D0021-fixed-FD-threshold-persistence",
        "fold": args.fold,
        "thresholds": [0.4, 0.5, 0.6],
        "aggregation": "arithmetic mean of the three scalar FD estimates",
        "threshold_scan": False,
        "fields": fields,
        "passed_fields": [
            field for field, result in fields.items() if result["passed"]
        ],
        "per_case": per_case,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
