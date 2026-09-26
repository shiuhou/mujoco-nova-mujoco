#!/usr/bin/env python3
"""Analyze measured Nova gripper calibration samples without changing a model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml


def _finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def analyze(record: dict) -> str:
    samples = record.get("samples") or []
    rows = []
    for sample in samples:
        command = _finite(sample.get("command"))
        left = _finite(sample.get("left_position_mm"))
        right = _finite(sample.get("right_position_mm"))
        gap = _finite(sample.get("gap_mm"))
        if command is not None and left is not None and right is not None:
            rows.append((command, left, right, gap))

    lines = ["Nova gripper calibration analysis", "", f"usable_samples: {len(rows)}"]
    if len(rows) < 2:
        lines += [
            "status: insufficient measured samples",
            "required: at least two samples with command, left_position_mm, and right_position_mm",
        ]
        return "\n".join(lines)

    rows.sort()
    commands = np.array([r[0] for r in rows])
    left = np.array([r[1] for r in rows])
    right = np.array([r[2] for r in rows])
    gaps = np.array([r[3] for r in rows if r[3] is not None])

    left_travel = float(left[-1] - left[0])
    right_travel = float(right[-1] - right[0])
    lines += [
        f"command_range: {commands[0]:.3f} .. {commands[-1]:.3f}",
        f"left_travel_mm: {left_travel:.6f}",
        f"right_travel_mm: {right_travel:.6f}",
        f"travel_difference_mm: {abs(abs(left_travel) - abs(right_travel)):.6f}",
    ]
    if len(gaps) >= 2:
        lines.append(f"total_gap_travel_mm: {float(gaps[-1] - gaps[0]):.6f}")
    else:
        lines.append("total_gap_travel_mm: unavailable (record gap_mm at >=2 samples)")

    left_fit = np.polyfit(commands, left, 1)
    right_fit = np.polyfit(commands, right, 1)
    left_pred = np.polyval(left_fit, commands)
    right_pred = np.polyval(right_fit, commands)
    left_residual = left - left_pred
    right_residual = right - right_pred
    symmetry = abs(abs(left_travel) - abs(right_travel))
    lines += [
        f"approximate_symmetry: {'yes' if symmetry <= 1.0 else 'no'} (threshold=1.0 mm)",
        "",
        "linear_fit_left_position_mm = "
        f"{left_fit[0]:.9g} * command + {left_fit[1]:.9g}",
        "linear_fit_right_position_mm = "
        f"{right_fit[0]:.9g} * command + {right_fit[1]:.9g}",
        f"left_fit_max_abs_residual_mm: {float(np.max(np.abs(left_residual))):.6f}",
        f"left_fit_rms_residual_mm: {float(np.sqrt(np.mean(left_residual**2))):.6f}",
        f"right_fit_max_abs_residual_mm: {float(np.max(np.abs(right_residual))):.6f}",
        f"right_fit_rms_residual_mm: {float(np.sqrt(np.mean(right_residual**2))):.6f}",
        "linearity_assessment: descriptive only; no MuJoCo parameter was changed",
        "",
        "per_sample:",
        "command left_mm right_mm gap_mm left_residual_mm right_residual_mm",
    ]
    for index, command in enumerate(commands):
        gap = rows[index][3]
        lines.append(
            f"{command:.3f} {left[index]:.6f} {right[index]:.6f} "
            f"{gap if gap is not None else float('nan'):.6f} "
            f"{left_residual[index]:.6f} {right_residual[index]:.6f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    with args.calibration.open(encoding="utf-8") as stream:
        record = yaml.safe_load(stream) or {}
    print(analyze(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
