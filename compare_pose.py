#!/usr/bin/env python3
"""Compare recorded Dobot GetAngle/GetPose samples with MuJoCo FK.

The comparison is offline. It accepts both the original one-record JSON
format and the Phase 1.5 dataset format with ``metadata`` and ``samples``.
For every sample it evaluates both the current and historical axis hypotheses
by default. Real Rx/Ry/Rz values are converted to rotation matrices before
orientation comparison; their components are never subtracted directly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from nova_fk import MODEL_PATH, TcpPose, tcp_pose


ORIENTATION_CONVENTIONS = frozenset({"rotation_vector", "rpy"})
AXIS_HYPOTHESES = ("current", "legacy")


@dataclass(frozen=True)
class SampleRecord:
    name: str
    q_deg: np.ndarray
    real_position_mm: np.ndarray
    real_rotation: np.ndarray
    orientation_convention: str
    user: int
    tool: int
    robot_mode: int | None = None
    error_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PoseComparison:
    sample: str
    axis_mode: str
    real_position_mm: np.ndarray
    simulated_position_mm: np.ndarray
    position_delta_mm: np.ndarray
    position_error_mm: float
    orientation_error_deg: float


def _rotation_vector_matrix(vector_deg: np.ndarray) -> np.ndarray:
    """Convert a Dobot rotation vector in degrees to a rotation matrix."""

    angle = math.radians(float(np.linalg.norm(vector_deg)))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector_deg / np.linalg.norm(vector_deg)
    x, y, z = axis
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def _rpy_matrix(values_deg: np.ndarray) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw values in degrees to a matrix."""

    roll, pitch, yaw = np.deg2rad(values_deg)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def _orientation_error_deg(
    real_rotation: np.ndarray, simulated_rotation: np.ndarray
) -> float:
    """Return the angle of ``R_err = R_real.T @ R_sim`` in degrees.

    Computing sine and cosine separately and combining them with ``atan2`` is
    stable both near zero and near a 180 degree relative rotation.
    """

    relative = real_rotation.T @ simulated_rotation
    sine = 0.5 * float(
        np.linalg.norm(
            np.array(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ]
            )
        )
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.atan2(max(sine, 0.0), cosine))


def _finite_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain exactly {length} finite values")
    return vector


def _parse_orientation(
    pose: np.ndarray, convention: Any, *, sample_name: str
) -> tuple[np.ndarray, str]:
    if convention not in ORIENTATION_CONVENTIONS:
        raise ValueError(
            f"sample {sample_name!r} must specify orientation as "
            "'rotation_vector' or 'rpy'"
        )
    matrix = (
        _rotation_vector_matrix(pose[3:])
        if convention == "rotation_vector"
        else _rpy_matrix(pose[3:])
    )
    return matrix, str(convention)


def _parse_sample(
    raw_sample: Any,
    *,
    index: int,
    metadata: dict[str, Any],
    root: dict[str, Any],
    orientation_override: str | None = None,
) -> SampleRecord:
    if not isinstance(raw_sample, dict):
        raise ValueError(f"sample {index} must be a JSON object")
    name = raw_sample.get("name", f"pose_{index + 1:02d}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"sample {index} has an invalid name")
    name = name.strip()

    user = raw_sample.get("user", metadata.get("user", root.get("user")))
    tool = raw_sample.get("tool", metadata.get("tool", root.get("tool")))
    if user != 0 or tool != 0:
        raise ValueError(f"sample {name!r} requires explicit user=0 and tool=0")

    q_values = raw_sample.get("q_deg", raw_sample.get("get_angle_deg"))
    pose_values = raw_sample.get(
        "pose_real_mm_deg",
        raw_sample.get("pose_mm_deg", raw_sample.get("get_pose")),
    )
    if q_values is None or pose_values is None:
        raise ValueError(
            f"sample {name!r} needs q_deg/get_angle_deg and "
            "pose_real_mm_deg/pose_mm_deg/get_pose"
        )
    q_deg = _finite_vector(q_values, length=6, label=f"sample {name!r} q_deg")
    pose = _finite_vector(pose_values, length=6, label=f"sample {name!r} pose_mm_deg")
    convention = orientation_override or raw_sample.get(
        "orientation",
        metadata.get(
            "orientation",
            metadata.get("pose_orientation_convention", root.get("orientation")),
        ),
    )
    real_rotation, convention = _parse_orientation(pose, convention, sample_name=name)

    robot_mode = raw_sample.get("robot_mode")
    if robot_mode is not None:
        if isinstance(robot_mode, bool) or not isinstance(robot_mode, (int, float)):
            raise ValueError(f"sample {name!r} has invalid robot_mode")
        if not float(robot_mode).is_integer():
            raise ValueError(f"sample {name!r} has non-integer robot_mode")
        robot_mode = int(robot_mode)

    error_values = raw_sample.get("error_ids", ())
    if not isinstance(error_values, (list, tuple)):
        raise ValueError(f"sample {name!r} error_ids must be an array")
    try:
        error_ids = tuple(int(value) for value in error_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {name!r} error_ids must be integers") from exc

    return SampleRecord(
        name=name,
        q_deg=q_deg,
        real_position_mm=pose[:3],
        real_rotation=real_rotation,
        orientation_convention=convention,
        user=0,
        tool=0,
        robot_mode=robot_mode,
        error_ids=error_ids,
    )


def load_samples(
    document: dict[str, Any], *, orientation_override: str | None = None
) -> list[SampleRecord]:
    """Validate and normalize either supported JSON dataset shape."""

    if not isinstance(document, dict):
        raise ValueError("JSON root must be an object")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if (
        orientation_override is not None
        and orientation_override not in ORIENTATION_CONVENTIONS
    ):
        raise ValueError("orientation_override must be 'rotation_vector' or 'rpy'")

    if "samples" in document:
        raw_samples = document["samples"]
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("samples must be a non-empty JSON array")
    else:
        raw_samples = [document]

    return [
        _parse_sample(
            raw_sample,
            index=index,
            metadata=metadata,
            root=document,
            orientation_override=orientation_override,
        )
        for index, raw_sample in enumerate(raw_samples)
    ]


def _record_values(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Backward-compatible single-record parser used by older callers."""

    sample = load_samples(record)
    if len(sample) != 1:
        raise ValueError("_record_values accepts exactly one sample")
    parsed = sample[0]
    return (
        parsed.q_deg,
        parsed.real_position_mm,
        parsed.real_rotation,
        parsed.orientation_convention,
    )


def compare_sample(
    sample: SampleRecord,
    *,
    model_path: Path = MODEL_PATH,
    axis_modes: Sequence[str] = AXIS_HYPOTHESES,
) -> list[PoseComparison]:
    comparisons: list[PoseComparison] = []
    for axis_mode in axis_modes:
        if axis_mode not in AXIS_HYPOTHESES:
            raise ValueError(f"unknown axis mode {axis_mode!r}")
        simulated: TcpPose = tcp_pose(
            np.deg2rad(sample.q_deg),
            model_path=model_path,
            axis_hypothesis=axis_mode,
        )
        simulated_position_mm = simulated.position_m * 1000.0
        delta_mm = simulated_position_mm - sample.real_position_mm
        comparisons.append(
            PoseComparison(
                sample=sample.name,
                axis_mode=axis_mode,
                real_position_mm=sample.real_position_mm,
                simulated_position_mm=simulated_position_mm,
                position_delta_mm=delta_mm,
                position_error_mm=float(np.linalg.norm(delta_mm)),
                orientation_error_deg=_orientation_error_deg(
                    sample.real_rotation, simulated.rotation_matrix
                ),
            )
        )
    return comparisons


def compare_samples(
    samples: Iterable[SampleRecord],
    *,
    model_path: Path = MODEL_PATH,
    axis_modes: Sequence[str] = AXIS_HYPOTHESES,
) -> list[PoseComparison]:
    return [
        comparison
        for sample in samples
        for comparison in compare_sample(
            sample, model_path=model_path, axis_modes=axis_modes
        )
    ]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
    }


def calculate_statistics(
    comparisons: Iterable[PoseComparison],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return position/orientation mean, median, RMS, and max by axis mode."""

    grouped: dict[str, list[PoseComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison.axis_mode, []).append(comparison)
    return {
        axis_mode: {
            "position_error_mm": _summary(
                np.asarray([item.position_error_mm for item in values])
            ),
            "orientation_error_deg": _summary(
                np.asarray([item.orientation_error_deg for item in values])
            ),
        }
        for axis_mode, values in grouped.items()
    }


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def diagnostic_messages(
    samples: Sequence[SampleRecord], comparisons: Sequence[PoseComparison]
) -> list[str]:
    """Describe patterns that may indicate frame, axis, or origin problems."""

    messages: list[str] = []
    statistics = calculate_statistics(comparisons)
    for axis_mode in AXIS_HYPOTHESES:
        if axis_mode not in statistics:
            continue
        position = statistics[axis_mode]["position_error_mm"]
        orientation = statistics[axis_mode]["orientation_error_deg"]
        messages.append(
            f"{axis_mode}: position RMS={position['rms']:.3f} mm, "
            f"orientation RMS={orientation['rms']:.3f} deg"
        )

    if all(axis in statistics for axis in AXIS_HYPOTHESES):
        current = statistics["current"]
        legacy = statistics["legacy"]
        current_position_rms = current["position_error_mm"]["rms"]
        legacy_position_rms = legacy["position_error_mm"]["rms"]
        current_orientation_rms = current["orientation_error_deg"]["rms"]
        legacy_orientation_rms = legacy["orientation_error_deg"]["rms"]
        if math.isclose(
            current_position_rms, legacy_position_rms, rel_tol=1e-6, abs_tol=1e-6
        ):
            messages.append("current and legacy position RMS are indistinguishable")
        else:
            winner = (
                "current" if current_position_rms < legacy_position_rms else "legacy"
            )
            messages.append(
                f"{winner} hypothesis has lower position RMS "
                f"({min(current_position_rms, legacy_position_rms):.3f} mm); "
                "this is evidence only"
            )
        if math.isclose(
            current_orientation_rms,
            legacy_orientation_rms,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            messages.append("current and legacy orientation RMS are indistinguishable")
        else:
            winner = (
                "current"
                if current_orientation_rms < legacy_orientation_rms
                else "legacy"
            )
            messages.append(
                f"{winner} hypothesis has lower orientation RMS "
                f"({min(current_orientation_rms, legacy_orientation_rms):.3f} deg); "
                "this is evidence only"
            )

    for axis_mode in sorted({item.axis_mode for item in comparisons}):
        mode_comparisons = [item for item in comparisons if item.axis_mode == axis_mode]
        deltas = np.asarray([item.position_delta_mm for item in mode_comparisons])
        mean_delta = np.mean(deltas, axis=0)
        residual_rms = float(
            np.sqrt(np.mean(np.sum(np.square(deltas - mean_delta), axis=1)))
        )
        mean_norm = float(np.linalg.norm(mean_delta))
        messages.append(
            f"{axis_mode}: mean position offset "
            f"[{mean_delta[0]:.3f}, {mean_delta[1]:.3f}, {mean_delta[2]:.3f}] mm; "
            f"residual RMS around offset={residual_rms:.3f} mm"
        )
        if mean_norm > 2.0 and residual_rms <= max(2.0, 0.2 * mean_norm):
            messages.append(
                f"{axis_mode}: approximately constant TCP/base/tool-frame offset is suspected; "
                "inspect frames before changing geometry"
            )
        elif residual_rms > max(2.0, 0.2 * mean_norm):
            messages.append(
                f"{axis_mode}: configuration-dependent position residual is present; "
                "inspect joint origins, axes, signs, and hierarchy"
            )

        if len(samples) >= 3:
            q = np.asarray([sample.q_deg for sample in samples])
            errors = np.asarray([item.position_error_mm for item in mode_comparisons])
            correlations = []
            for joint_index in range(6):
                correlation = _pearson_correlation(q[:, joint_index], errors)
                if correlation is not None:
                    correlations.append(
                        (abs(correlation), joint_index + 1, correlation)
                    )
            if correlations:
                _magnitude, joint_index, correlation = max(correlations)
                messages.append(
                    f"{axis_mode}: strongest position-error correlation is J{joint_index} "
                    f"(Pearson r={correlation:.3f}); correlation is diagnostic only"
                )
    return messages


def _print_comparisons(
    comparisons: Sequence[PoseComparison],
) -> None:
    print("Per-sample comparison")
    print("sample axis_mode pos_err_mm ori_err_deg dx dy dz")
    for comparison in comparisons:
        dx, dy, dz = comparison.position_delta_mm
        print(
            f"{comparison.sample} {comparison.axis_mode} "
            f"{comparison.position_error_mm:.6f} "
            f"{comparison.orientation_error_deg:.6f} "
            f"{dx:.6f} {dy:.6f} {dz:.6f}"
        )
        print(
            "  real_position_mm: "
            f"{np.array2string(comparison.real_position_mm, precision=6)}"
        )
        print(
            "  simulated_position_mm: "
            f"{np.array2string(comparison.simulated_position_mm, precision=6)}"
        )
    print()


def _print_statistics(comparisons: Sequence[PoseComparison]) -> None:
    print("Aggregate statistics")
    statistics = calculate_statistics(comparisons)
    for axis_mode in AXIS_HYPOTHESES:
        if axis_mode not in statistics:
            continue
        print(f"axis_mode: {axis_mode}")
        for metric, values in statistics[axis_mode].items():
            print(
                f"  {metric}: mean={values['mean']:.6f}, "
                f"median={values['median']:.6f}, rms={values['rms']:.6f}, "
                f"max={values['max']:.6f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="offline JSON sample dataset")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--axis-hypothesis",
        choices=("current", "legacy", "both"),
        default="both",
        help="default is both; explicit modes remain for focused reports",
    )
    parser.add_argument(
        "--orientation",
        choices=("metadata", "rotation_vector", "rpy"),
        default="metadata",
        help=(
            "orientation convention override; metadata is the default. "
            "Use rpy for the observed Dobot GetPose convention."
        ),
    )
    args = parser.parse_args()

    try:
        document = json.loads(args.record.read_text(encoding="utf-8"))
        samples = load_samples(
            document,
            orientation_override=(
                None if args.orientation == "metadata" else args.orientation
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"cannot load comparison dataset: {exc}") from exc

    axis_modes = (
        AXIS_HYPOTHESES if args.axis_hypothesis == "both" else (args.axis_hypothesis,)
    )
    comparisons = compare_samples(samples, model_path=args.model, axis_modes=axis_modes)
    print(f"samples: {len(samples)}")
    conventions = sorted({sample.orientation_convention for sample in samples})
    print(f"orientation_conventions: {', '.join(conventions)}")
    print("orientation_error: angle of R_err = R_real.T @ R_sim")
    _print_comparisons(comparisons)
    _print_statistics(comparisons)
    print("Diagnostics")
    for message in diagnostic_messages(samples, comparisons):
        print(f"- {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
