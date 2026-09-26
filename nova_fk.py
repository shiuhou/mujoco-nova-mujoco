#!/usr/bin/env python3
"""Offline Nova2 MuJoCo FK and Dobot pose formatting.

This module never opens a network connection. It only loads the local MJCF,
sets six qpos values, and evaluates MuJoCo forward kinematics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
    raise SystemExit(
        "MuJoCo Python bindings are required. Install with: python -m pip install mujoco"
    ) from exc


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "nova2.xml"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
LEGACY_AXIS_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class TcpPose:
    position_m: np.ndarray
    world_position_m: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_wxyz: np.ndarray
    rotation_vector_deg: np.ndarray


def _rotation_vector_deg(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a mathematical rotation vector."""
    trace = float(np.trace(matrix))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.zeros(3)
    if math.pi - angle < 1e-7:
        # At pi the skew part vanishes; recover a stable axis from the diagonal.
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != pivot and axis[pivot] > 1e-10:
                    axis[index] = matrix[pivot, index] / (2.0 * axis[pivot])
            axis /= np.linalg.norm(axis)
        return axis * math.degrees(angle)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    axis /= 2.0 * math.sin(angle)
    return axis * math.degrees(angle)


def _quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to MuJoCo-style wxyz quaternion."""
    q = np.empty(4)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        q[0] = 0.25 * scale
        q[1] = (matrix[2, 1] - matrix[1, 2]) / scale
        q[2] = (matrix[0, 2] - matrix[2, 0]) / scale
        q[3] = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * math.sqrt(
            max(
                1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index], 0.0
            )
        )
        q[index + 1] = 0.25 * scale
        q[0] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        q[next_index + 1] = (
            matrix[next_index, index] + matrix[index, next_index]
        ) / scale
        q[last_index + 1] = (
            matrix[last_index, index] + matrix[index, last_index]
        ) / scale
    return q / np.linalg.norm(q)


def load_model(model_path: Path = MODEL_PATH):
    """Load the standalone MJCF model."""
    return mujoco.MjModel.from_xml_path(str(model_path))


def tcp_pose(
    q_rad: Sequence[float],
    *,
    model_path: Path = MODEL_PATH,
    axis_hypothesis: str = "current",
) -> TcpPose:
    """Evaluate TCP pose for six joint angles in radians.

    ``legacy`` reproduces the pre-2026 source revision convention where J1,
    J4, and J5 used ``-Z``. It is represented by negating those q coordinates
    because the current MJCF stores the official pinned ``+Z`` axes.
    """
    q = np.asarray(tuple(q_rad), dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("q_rad must contain six finite values")
    if axis_hypothesis == "legacy":
        q = q * LEGACY_AXIS_SIGN
    elif axis_hypothesis != "current":
        raise ValueError("axis_hypothesis must be 'current' or 'legacy'")

    model = load_model(model_path)
    data = mujoco.MjData(model)
    for name, value in zip(JOINT_NAMES, q, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MJCF is missing joint {name!r}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if site_id < 0:
        raise RuntimeError("MJCF is missing site 'tcp'")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise RuntimeError("MJCF is missing body 'base_link'")
    world_position = data.site_xpos[site_id].copy()
    base_rotation = data.xmat[base_id].reshape(3, 3).copy()
    # The source description adds world_joint z=0.03 for the mounted scene.
    # GetPose(user=0, tool=0) is compared in the robot base frame.
    position = base_rotation.T @ (world_position - data.xpos[base_id])
    rotation = base_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
    return TcpPose(
        position_m=position,
        world_position_m=world_position,
        rotation_matrix=rotation,
        quaternion_wxyz=_quaternion_wxyz(rotation),
        rotation_vector_deg=_rotation_vector_deg(rotation),
    )


def _parse_q(args: argparse.Namespace) -> np.ndarray:
    if args.q_rad is not None and args.q_deg is not None:
        raise SystemExit("choose only one of --q-rad or --q-deg")
    values = args.q_rad if args.q_rad is not None else args.q_deg
    if values is None:
        values = [0.0] * 6
    q = np.asarray(values, dtype=float)
    return q if args.q_rad is not None else np.deg2rad(q)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-rad", nargs=6, type=float, metavar="Q", help="joint angles in radians"
    )
    parser.add_argument(
        "--q-deg", nargs=6, type=float, metavar="Q", help="joint angles in degrees"
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--axis-hypothesis", choices=("current", "legacy"), default="current"
    )
    args = parser.parse_args()
    q = _parse_q(args)
    pose = tcp_pose(q, model_path=args.model, axis_hypothesis=args.axis_hypothesis)
    print("q_rad:", np.array2string(q, precision=9))
    print("axis_hypothesis:", args.axis_hypothesis)
    print("tcp_position_m:", np.array2string(pose.position_m, precision=9))
    print("tcp_position_mm:", np.array2string(pose.position_m * 1000.0, precision=6))
    print("tcp_position_world_m:", np.array2string(pose.world_position_m, precision=9))
    print("tcp_quaternion_wxyz:", np.array2string(pose.quaternion_wxyz, precision=9))
    print(
        "tcp_rotation_vector_deg:",
        np.array2string(pose.rotation_vector_deg, precision=6),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
