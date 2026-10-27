"""Offline, fixed-seed FK/IK audit. SciPy is an audit-only independent numerical reference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cartesian_bridge import CartesianIKSolver, IKConfig
from nova_simulator import NovaMujocoSimulator, Pose


class UrdfReference:
    """Serial homogeneous transforms parsed directly from pinned vendor XML.

    Independent evaluation implementation, NOT independent physical geometry.
    No xacro execution: this pinned description's six origins are numeric.
    """

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = []
        parent = "base_link"
        for i in range(1, 7):
            joint = root.find(f"joint[@name='joint{i}']")
            if joint is None or joint.find("parent").get("link") != parent:
                raise ValueError("expected contiguous Nova2 serial chain")
            parent = joint.find("child").get("link")
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy"), sep=" ")
            ).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            limit = joint.find("limit")
            self.joints.append(
                (
                    transform,
                    axis,
                    [float(limit.get("lower")), float(limit.get("upper"))],
                )
            )
        self.limits = np.array([j[2] for j in self.joints])

    def pose(self, q):
        matrix = np.eye(4)
        for angle, (origin, axis, _) in zip(q, self.joints, strict=True):
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            matrix = matrix @ origin @ turn
        return Pose(matrix[:3, 3].copy(), matrix[:3, :3].copy())


def residual(target, actual):
    return np.r_[
        target.position_m - actual.position_m,
        Rotation.from_matrix(
            target.rotation_matrix @ actual.rotation_matrix.T
        ).as_rotvec(),
    ]


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(
        zip(
            ("median", "p95", "p99", "max"),
            map(float, np.percentile(a, [50, 95, 99, 100])),
        )
    ) | {"rms": float(np.sqrt(np.mean(a * a)))}


def evaluate(sim, target, seed):
    start = time.perf_counter_ns()
    result = CartesianIKSolver(sim).solve(target, seed_q=seed)
    latency = (time.perf_counter_ns() - start) / 1e6
    original = sim.get_joint_position()
    sim.set_joint_state_for_kinematics(result.q_rad)
    err = residual(target, sim.get_tool_pose())
    sim.set_joint_state_for_kinematics(original)
    return {
        "status": result.status,
        "latency_ms": latency,
        "position_mm": float(np.linalg.norm(err[:3]) * 1000),
        "orientation_rad": float(np.linalg.norm(err[3:])),
        "within_limits": bool(
            np.all(result.q_rad >= sim.joint_limits[:, 0])
            and np.all(result.q_rad <= sim.joint_limits[:, 1])
        ),
        "seed_distance_rad": float(np.linalg.norm(result.q_rad - seed)),
        "q_rad": result.q_rad.tolist(),
    }


def summarize(rows):
    success = [r for r in rows if r["status"] == "converged"]
    return {
        "count": len(rows),
        "converged": len(success),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "limit_violations": sum(not r["within_limits"] for r in rows),
        "latency_ms_all": distribution([r["latency_ms"] for r in rows]),
        "position_mm_success": distribution([r["position_mm"] for r in success]),
        "orientation_rad_success": distribution(
            [r["orientation_rad"] for r in success]
        ),
        "position_mm_all": distribution([r["position_mm"] for r in rows]),
        "orientation_rad_all": distribution([r["orientation_rad"] for r in rows]),
    }


def run(urdf, count=256, seed=20261027):
    rng = np.random.default_rng(seed)
    ref = UrdfReference(urdf)
    sim = NovaMujocoSimulator()
    low, high = sim.joint_limits.T
    if not np.array_equal(sim.joint_limits, ref.limits):
        raise ValueError("URDF/MJCF limit disagreement")
    qs = rng.uniform(low, high, (count, 6))
    edges = np.array(
        [
            np.where(
                np.array([(i >> j) & 1 for j in range(6)]), high - 1e-7, low + 1e-7
            )
            for i in range(64)
        ]
    )
    singular = rng.uniform(low * 0.7, high * 0.7, (32, 6))
    singular[:, 2] = 0
    singular[:, 4] = np.tile([0, 1e-8, np.pi, -np.pi], 8)
    all_q = np.vstack([qs, edges, singular, np.zeros((1, 6))])
    fk_errors = []
    min_singular = []
    poses = []
    for q in all_q:
        sim.set_joint_state_for_kinematics(q)
        pose = sim.get_tool_pose()
        poses.append(pose)
        fk_errors.append(residual(ref.pose(q), pose))
        min_singular.append(
            float(np.linalg.svd(np.vstack(sim.site_jacobian()), compute_uv=False)[-1])
        )
    fk_errors = np.array(fk_errors)
    groups = {}
    for name, targets in [("random", qs), ("limits", edges), ("singular", singular)]:
        for strategy in ("near", "zero"):
            rows = []
            for q in targets:
                sim.set_joint_state_for_kinematics(q)
                target = sim.get_tool_pose()
                initial = (
                    np.clip(q + rng.normal(0, 0.05, 6), low, high)
                    if strategy == "near"
                    else np.zeros(6)
                )
                sim.reset(initial)
                rows.append(evaluate(sim, target, initial))
            groups[name + "_" + strategy] = {"summary": summarize(rows), "rows": rows}
    # Established bounded optimizer, independent URDF FK and finite-difference Jacobian.
    # Same targets/seeds/tolerances, but different iteration/evaluation budgets.
    comparison = []
    for q in qs[: min(count, 32)]:
        target = ref.pose(q)
        sim.reset(np.zeros(6))
        production = evaluate(sim, target, np.zeros(6))
        start = time.perf_counter_ns()
        fit = least_squares(
            lambda x: residual(target, ref.pose(x)),
            np.zeros(6),
            bounds=(low, high),
            max_nfev=200,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        ms = (time.perf_counter_ns() - start) / 1e6
        err = residual(target, ref.pose(fit.x))
        comparison.append(
            {
                "production": production,
                "scipy": {
                    "status": "converged"
                    if np.linalg.norm(err[:3]) <= 1e-6
                    and np.linalg.norm(err[3:]) <= 1e-6
                    else "residual_rejected",
                    "position_mm": float(np.linalg.norm(err[:3]) * 1000),
                    "orientation_rad": float(np.linalg.norm(err[3:])),
                    "latency_ms": ms,
                    "within_limits": bool(
                        np.all(fit.x >= low) and np.all(fit.x <= high)
                    ),
                    "nfev": fit.nfev,
                },
            }
        )
    # Smooth reachable joint-derived trajectory, including a wrist singularity crossing.
    trajectories = []
    for crossing in (False, True):
        center = np.array([0.2, -0.6, 1.4, 0.2, 0 if crossing else -1.2, -0.5])
        previous = center.copy()
        rows, jumps = [], []
        for t in np.linspace(-1, 1, 101):
            q = center + np.array(
                [0.03 * np.sin(t), 0.02 * t, -0.03 * t, 0.02 * t, 0.1 * t, -0.02 * t]
            )
            sim.set_joint_state_for_kinematics(q)
            target = sim.get_tool_pose()
            sim.reset(previous)
            row = evaluate(sim, target, previous)
            rows.append(row)
            if row["status"] != "converged":
                break
            solved = np.array(row["q_rad"])
            jumps.append(float(np.max(np.abs(solved - previous))))
            previous = solved
        trajectories.append(
            {
                "wrist_singularity_crossing": crossing,
                "summary": summarize(rows),
                "max_joint_step_rad": max(jumps, default=None),
                "initial_seed_to_first_solution_rad": jumps[0] if jumps else None,
                "adjacent_waypoint_joint_step_rad": distribution(jumps[1:]),
            }
        )
    # Rotating world mounting must not change base-coordinate IK.
    quat = Rotation.from_euler("xyz", [0.4, -0.6, 1.2]).as_quat()
    sim.model.body_quat[sim._base_id] = quat[[3, 0, 1, 2]]
    sim.reset(np.array([0.2, -0.6, 1.4, 0.2, -1.2, -0.5]))
    pose = sim.get_tool_pose()
    rotated = evaluate(
        sim,
        Pose(pose.position_m + [0.005, 0, 0], pose.rotation_matrix),
        sim.get_joint_position(),
    )
    unreachable = evaluate(
        sim, Pose(np.array([10.0, 10.0, 10.0]), np.eye(3)), sim.get_joint_position()
    )
    return {
        "schema": 1,
        "seed": seed,
        "random_count": count,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
        },
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            p: hashlib.sha256(
                (Path(__file__) if p == "kinematics_audit.py" else Path(p)).read_bytes()
            ).hexdigest()
            for p in [
                "nova2.xml",
                "nova_simulator.py",
                "cartesian_bridge.py",
                "kinematics_audit.py",
            ]
        },
        "urdf_sha256": hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
        "scope": "offline mathematical consistency only; no hardware/network; sampled reachability ignores collisions",
        "tolerances": {
            "position_m": IKConfig().position_tolerance_m,
            "orientation_rad": IKConfig().orientation_tolerance_rad,
        },
        "fk": {
            "count": len(all_q),
            "position_mm": distribution(
                np.linalg.norm(fk_errors[:, :3], axis=1) * 1000
            ),
            "orientation_rad": distribution(np.linalg.norm(fk_errors[:, 3:], axis=1)),
            "jacobian_smallest_singular_value": distribution(min_singular),
            "minimum_jacobian_singular_value": min(min_singular),
        },
        "ik": groups,
        "independent_solver": {
            "production": summarize([r["production"] for r in comparison]),
            "scipy": summarize([r["scipy"] for r in comparison]),
            "rows": comparison,
        },
        "trajectories": trajectories,
        "rotated_base": rotated,
        "unreachable": unreachable,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.count <= 4096:
        parser.error("count must be 1..4096")
    args.output.mkdir(parents=True, exist_ok=False)
    report = run(args.urdf, args.count)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("ik", "independent_solver")},
            indent=2,
        )
    )
    for name, group in report["ik"].items():
        print(name, json.dumps(group["summary"]))
    print(
        "independent_solver",
        json.dumps(
            {k: v for k, v in report["independent_solver"].items() if k != "rows"}
        ),
    )


if __name__ == "__main__":
    main()
