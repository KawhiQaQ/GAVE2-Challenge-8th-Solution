#!/usr/bin/env python3
"""Build the released SIVA-style C-zone masks from optic-disc masks.

The region is the annulus from 1.5 to 2.5 optic-disc diameters around the
minimum-enclosing-circle center. Outputs are a cache derived only from image
geometry; no A/V labels or biomarker values are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zone_c_from_disc(disc: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    _, binary = cv2.threshold(disc, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("optic-disc mask has no foreground component")
    contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    diameter = 2.0 * radius
    center = (int(center_x), int(center_y))
    inner_radius = int(1.5 * diameter)
    outer_radius = int(2.5 * diameter)
    zone = np.zeros_like(disc, dtype=np.uint8)
    cv2.circle(zone, center, outer_radius, 255, -1)
    cv2.circle(zone, center, inner_radius, 0, -1)
    return zone, {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "disc_diameter": float(diameter),
        "inner_radius": float(inner_radius),
        "outer_radius": float(outer_radius),
        "region_pixels": float(np.count_nonzero(zone)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disc-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-output")
    args = parser.parse_args()

    disc_dir = Path(args.disc_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    files = sorted(disc_dir.glob("g_*.png"))
    if not files:
        raise FileNotFoundError(f"No g_*.png masks in {disc_dir}")
    output_dir.mkdir(parents=True)

    cases: dict[str, object] = {}
    for source in files:
        disc = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if disc is None:
            raise FileNotFoundError(source)
        zone, diagnostics = zone_c_from_disc(disc)
        destination = output_dir / source.name
        if not cv2.imwrite(str(destination), zone):
            raise OSError(f"Could not write {destination}")
        cases[source.stem] = {
            **diagnostics,
            "source_sha256": file_sha256(source),
            "output_sha256": file_sha256(destination),
        }

    report = {
        "schema_version": 1,
        "source": str(disc_dir),
        "output": str(output_dir),
        "definition": "annulus_1.5DD_to_2.5DD",
        "case_count": len(files),
        "cases": cases,
    }
    report_path = (
        Path(args.report_output).resolve()
        if args.report_output
        else output_dir / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "case_count": len(files),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
