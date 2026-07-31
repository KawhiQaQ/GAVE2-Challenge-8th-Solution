#!/usr/bin/env python3
"""Fit and apply a low-degree-of-freedom Task3 density calibration.

The calibration is deliberately narrow:

* one multiplicative factor for ``artery_density``;
* one multiplicative factor for ``vein_density``;
* all other Task3 values and every Task1/Task2 byte remain unchanged.

Factors are fitted from genuine out-of-fold predictions. ``fit-evaluate`` also
reports leave-one-out performance so the case being evaluated never contributes
to its own factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


DENSITY_KEYS = ("artery_density", "vein_density")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = (np.abs(truth) + np.abs(prediction)) / 2.0
    terms = np.divide(
        np.abs(truth - prediction),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(100.0 * terms.mean())


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(truth - prediction).mean()),
        "smape_percent": _smape(truth, prediction),
    }


def _mean_ratio_factor(truth: np.ndarray, prediction: np.ndarray) -> float:
    prediction_mean = float(prediction.mean())
    if prediction_mean <= 0:
        raise ValueError("Prediction mean must be positive for ratio calibration")
    return float(truth.mean() / prediction_mean)


def _fit_evaluate(args: argparse.Namespace) -> None:
    per_case_path = Path(args.per_case)
    raw = json.loads(per_case_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) < 3:
        raise ValueError("Expected a per-case object with at least three cases")

    case_ids = sorted(raw)
    calibration: dict[str, object] = {
        "schema_version": 1,
        "method": "mean_truth_over_mean_prediction",
        "source": str(per_case_path),
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "factors": {},
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "leave_one_out_mean_ratio",
        "source": str(per_case_path),
        "n_cases": len(case_ids),
        "metrics": {},
    }

    for key in DENSITY_KEYS:
        truth = np.asarray(
            [raw[case_id]["truth"][key] for case_id in case_ids],
            dtype=np.float64,
        )
        prediction = np.asarray(
            [raw[case_id]["prediction"][key] for case_id in case_ids],
            dtype=np.float64,
        )
        factor = _mean_ratio_factor(truth, prediction)
        loo_prediction = np.empty_like(prediction)
        loo_factors: list[float] = []
        for held_out in range(len(case_ids)):
            training_mask = np.arange(len(case_ids)) != held_out
            held_out_factor = _mean_ratio_factor(
                truth[training_mask],
                prediction[training_mask],
            )
            loo_factors.append(held_out_factor)
            loo_prediction[held_out] = prediction[held_out] * held_out_factor

        raw_metrics = _metrics(truth, prediction)
        loo_metrics = _metrics(truth, loo_prediction)
        calibration["factors"][key] = factor
        report["metrics"][key] = {
            "truth_mean": float(truth.mean()),
            "prediction_mean": float(prediction.mean()),
            "full_oof_factor": factor,
            "loo_factor_min": float(min(loo_factors)),
            "loo_factor_max": float(max(loo_factors)),
            "raw": raw_metrics,
            "leave_one_out_calibrated": loo_metrics,
            "mae_relative_reduction": float(
                1.0 - loo_metrics["mae"] / raw_metrics["mae"]
            ),
            "smape_relative_reduction": float(
                1.0
                - loo_metrics["smape_percent"] / raw_metrics["smape_percent"]
            ),
        }

    calibration_output = Path(args.calibration_output)
    report_output = Path(args.report_output)
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"calibration": calibration, "report": report}, indent=2))


def _replace_density_lines(
    content: bytes,
    factors: dict[str, float],
) -> tuple[bytes, dict[str, tuple[float, float]]]:
    text = content.decode("utf-8")
    lines = text.splitlines()
    replacements: dict[str, tuple[float, float]] = {}
    output_lines: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] in factors:
            key = fields[0]
            if key in replacements:
                raise ValueError(f"Duplicate {key} line")
            original = float(fields[1])
            calibrated = original * factors[key]
            replacements[key] = (original, calibrated)
            output_lines.append(f"{key} {calibrated:.6f}")
        else:
            output_lines.append(line)
    missing = set(factors) - set(replacements)
    if missing:
        raise ValueError(f"Missing density keys: {sorted(missing)}")
    trailing_newline = "\n" if text.endswith("\n") else ""
    return ("\n".join(output_lines) + trailing_newline).encode("utf-8"), replacements


def _apply_zip(args: argparse.Namespace) -> None:
    input_path = Path(args.input_zip)
    output_path = Path(args.output_zip)
    calibration_path = Path(args.calibration)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    factors = {
        key: float(calibration["factors"][key])
        for key in DENSITY_KEYS
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changed_cases: dict[str, dict[str, tuple[float, float]]] = {}
    copied_entry_count = 0

    with ZipFile(input_path, "r") as source, ZipFile(
        output_path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as target:
        for info in source.infolist():
            content = source.read(info.filename)
            parts = Path(info.filename).parts
            is_task3_txt = (
                len(parts) >= 2
                and parts[-2] == "Task3"
                and parts[-1].endswith(".txt")
            )
            if is_task3_txt:
                updated, replacements = _replace_density_lines(content, factors)
                changed_cases[Path(info.filename).stem] = replacements
                content = updated
            else:
                copied_entry_count += 1
            target.writestr(
                info,
                content,
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )

    with ZipFile(output_path, "r") as result:
        bad_entry = result.testzip()
        entry_count = len(result.infolist())
    if bad_entry is not None:
        raise RuntimeError(f"CRC failure in {bad_entry}")
    if len(changed_cases) != args.expected_cases:
        raise ValueError(
            f"Expected {args.expected_cases} Task3 files, found {len(changed_cases)}"
        )

    application_report = {
        "schema_version": 1,
        "input_zip": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_zip": str(output_path),
        "output_sha256": _sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "entry_count": entry_count,
        "changed_task3_cases": len(changed_cases),
        "unchanged_entries_copied": copied_entry_count,
        "factors": factors,
        "crc_ok": True,
        "changed_cases": changed_cases,
    }
    if args.report_output:
        report_output = Path(args.report_output)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            json.dumps(application_report, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(application_report, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit-evaluate",
        help="Fit full-OOF factors and evaluate them with leave-one-out.",
    )
    fit_parser.add_argument("--per-case", required=True)
    fit_parser.add_argument("--calibration-output", required=True)
    fit_parser.add_argument("--report-output", required=True)
    fit_parser.set_defaults(func=_fit_evaluate)

    apply_parser = subparsers.add_parser(
        "apply-zip",
        help="Apply fitted density factors to Task3 TXT files in a submission ZIP.",
    )
    apply_parser.add_argument("--input-zip", required=True)
    apply_parser.add_argument("--output-zip", required=True)
    apply_parser.add_argument("--calibration", required=True)
    apply_parser.add_argument("--report-output")
    apply_parser.add_argument("--expected-cases", type=int, default=50)
    apply_parser.add_argument("--overwrite", action="store_true")
    apply_parser.set_defaults(func=_apply_zip)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
