#!/usr/bin/env python3
"""Open an interactive, offline MuJoCo viewer for the Nova2 model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo is not installed in this Python. Use run_visualize.sh, or install "
        "it with: python -m pip install mujoco"
    ) from exc

from nova_fk import JOINT_NAMES, LEGACY_AXIS_SIGN, MODEL_PATH


def _joint_values(args: argparse.Namespace) -> np.ndarray:
    if args.q_rad is not None and args.q_deg is not None:
        raise SystemExit("choose only one of --q-rad or --q-deg")
    values = args.q_rad if args.q_rad is not None else args.q_deg
    q = np.zeros(6) if values is None else np.asarray(values, dtype=float)
    if args.q_deg is not None:
        q = np.deg2rad(q)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise SystemExit("six finite joint values are required")
    if args.axis_hypothesis == "legacy":
        q = q * LEGACY_AXIS_SIGN
    return q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-rad", nargs=6, type=float, metavar="Q")
    parser.add_argument("--q-deg", nargs=6, type=float, metavar="Q")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--axis-hypothesis", choices=("current", "legacy"), default="current"
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="also show the simple Phase 1 collision boxes",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    q = _joint_values(args)
    for name, value in zip(JOINT_NAMES, q, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise SystemExit(f"model is missing {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    print("Nova2 MuJoCo viewer (offline; no hardware connection)")
    print("q_rad:", np.array2string(q, precision=6))
    print("Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as handle:
        # Collision proxies are intentionally hidden by default so the
        # official STL visual geometry is not obscured by boxes.
        if not args.show_collision:
            handle.opt.geomgroup[1] = 0
        handle.cam.lookat[:] = [0.0, 0.0, 0.45]
        handle.cam.distance = 1.25
        handle.cam.azimuth = 135.0
        handle.cam.elevation = -18.0
        while handle.is_running():
            handle.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
