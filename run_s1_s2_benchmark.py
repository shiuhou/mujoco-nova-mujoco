#!/usr/bin/env python3
"""Run the reproducible offline Nova2 S1/S2 action-interface benchmark."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cartesian_bridge import NovaMujocoChunkExecutor, rotation_distance
from nova_simulator import NovaMujocoSimulator, SimulatorConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "evidence" / "sim_s1_s2"
VALIDATED_START_NAMES = (
    "pose_01",
    "pose_04",
    "pose_06",
    "pose_07",
    "pose_08",
    "pose_10",
    "pose_11",
    "pose_12",
)
DIRECTIONS = {
    "+X": np.array([0.010, 0.0, 0.0]),
    "-X": np.array([-0.010, 0.0, 0.0]),
    "+Y": np.array([0.0, 0.010, 0.0]),
    "-Y": np.array([0.0, -0.010, 0.0]),
    "+Z": np.array([0.0, 0.0, 0.010]),
    "-Z": np.array([0.0, 0.0, -0.010]),
}


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_starts() -> dict[str, np.ndarray]:
    document = json.loads((ROOT / "real_nova2_samples.json").read_text())
    samples = {sample["name"]: sample for sample in document["samples"]}
    missing = set(VALIDATED_START_NAMES) - set(samples)
    if missing:
        raise RuntimeError(f"real sample dataset is missing starts: {sorted(missing)}")
    return {
        name: np.deg2rad(np.asarray(samples[name]["q_deg"], dtype=np.float64))
        for name in VALIDATED_START_NAMES
    }


def _action(position: np.ndarray) -> np.ndarray:
    return np.array([*position, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _run_case(
    start_name: str,
    q_start: np.ndarray,
    direction: str,
    delta: np.ndarray,
) -> dict[str, Any]:
    simulator = NovaMujocoSimulator()
    simulator.reset(q_start)
    start_pose = simulator.get_link6_pose()
    target_position = start_pose.position_m + delta
    executor = NovaMujocoChunkExecutor(simulator)
    result = executor.execute_chunk([_action(target_position)])
    record = result.records[0] if result.records else None
    achieved_pose = simulator.get_link6_pose()
    actual_delta = achieved_pose.position_m - start_pose.position_m
    position_residual = achieved_pose.position_m - target_position
    orientation_residual = rotation_distance(
        achieved_pose.rotation_matrix, start_pose.rotation_matrix
    )
    passed = (
        result.succeeded
        and float(np.linalg.norm(position_residual))
        <= executor.ik.config.position_tolerance_m
        and orientation_residual <= executor.ik.config.orientation_tolerance_rad
    )
    return {
        "start_name": start_name,
        "direction": direction,
        "mode": simulator.mode,
        "action_frame": "base",
        "controlled_frame": executor.frame,
        "q_start_rad": q_start.tolist(),
        "pose_start": start_pose.as_dict(),
        "requested_action": _action(target_position).tolist(),
        "target_pose": {
            "position_m": target_position.tolist(),
            "rotation_matrix": start_pose.rotation_matrix.tolist(),
        },
        "execution_status": result.status,
        "failure_reason": result.reason,
        "ik": None if record is None else record.ik.as_dict(),
        "achieved_q_rad": simulator.get_joint_position().tolist(),
        "achieved_pose": achieved_pose.as_dict(),
        "requested_delta_m": delta.tolist(),
        "actual_delta_m": actual_delta.tolist(),
        "position_residual_m": position_residual.tolist(),
        "position_residual_norm_m": float(np.linalg.norm(position_residual)),
        "orientation_residual_rad": orientation_residual,
        "orientation_residual_deg": math.degrees(orientation_residual),
        "contacts": [] if record is None else list(record.contacts),
        "contact_count": 0 if record is None else len(record.contacts),
        "simulation_time_s": simulator.simulation_time_s,
        "pass": passed,
    }


def _simulation_throughput(steps: int = 50_000) -> dict[str, float | int]:
    simulator = NovaMujocoSimulator()
    started = time.perf_counter()
    simulator.step(steps)
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "simulation_time_s": simulator.simulation_time_s,
        "host_elapsed_s": elapsed,
        "steps_per_host_second": steps / elapsed,
    }


def _summary(cases: list[dict[str, Any]], host_elapsed_s: float) -> dict[str, Any]:
    passed = [case for case in cases if case["pass"]]
    errors_mm = np.array([case["position_residual_norm_m"] * 1000 for case in cases])
    orientation_deg = np.array([case["orientation_residual_deg"] for case in cases])
    iterations = np.array(
        [case["ik"]["iterations"] for case in cases if case["ik"] is not None],
        dtype=float,
    )
    displacement = np.array(
        [
            case["ik"]["joint_displacement_rad"]
            for case in cases
            if case["ik"] is not None
        ]
    )
    return {
        "case_count": len(cases),
        "pass_count": len(passed),
        "success_rate": len(passed) / len(cases),
        "position_error_mm": {
            "mean": float(np.mean(errors_mm)),
            "rms": float(np.sqrt(np.mean(errors_mm**2))),
            "max": float(np.max(errors_mm)),
        },
        "orientation_residual_deg": {
            "mean": float(np.mean(orientation_deg)),
            "rms": float(np.sqrt(np.mean(orientation_deg**2))),
            "max": float(np.max(orientation_deg)),
        },
        "joint_displacement_rad": {
            "mean": float(np.mean(displacement)),
            "max": float(np.max(displacement)),
        },
        "ik_iterations": {
            "mean": float(np.mean(iterations)),
            "max": int(np.max(iterations)),
        },
        "contact_case_count": sum(case["contact_count"] > 0 for case in cases),
        "host_elapsed_s": host_elapsed_s,
        "ik_solves_per_host_second": len(cases) / host_elapsed_s,
    }


def _markdown(document: dict[str, Any]) -> str:
    summary = document["summary"]
    throughput = document["performance"]["simulation"]
    lines = [
        "# Nova2 MuJoCo S1/S2 Offline Benchmark",
        "",
        f"Generated: `{document['generated_at_utc']}`",
        f"Simulator revision: `{document['simulator_revision']}`",
        "",
        "This is a simulator action-interface characterization. It is not a real Nova",
        "controller-accuracy, dynamics, collision-safety, or hardware validation claim.",
        "",
        "## Aggregate",
        "",
        f"- Cases: {summary['pass_count']}/{summary['case_count']} PASS "
        f"({summary['success_rate']:.1%})",
        f"- Position residual RMS/max: {summary['position_error_mm']['rms']:.6f} / "
        f"{summary['position_error_mm']['max']:.6f} mm",
        f"- Orientation residual RMS/max: "
        f"{summary['orientation_residual_deg']['rms']:.9f} / "
        f"{summary['orientation_residual_deg']['max']:.9f} deg",
        f"- IK iterations mean/max: {summary['ik_iterations']['mean']:.3f} / "
        f"{summary['ik_iterations']['max']}",
        f"- Joint displacement mean/max: {summary['joint_displacement_rad']['mean']:.6f} / "
        f"{summary['joint_displacement_rad']['max']:.6f} rad",
        f"- Cases reporting coarse-proxy contacts: {summary['contact_case_count']}",
        f"- IK solves per host second: {summary['ik_solves_per_host_second']:.1f}",
        f"- MuJoCo forward steps per host second: {throughput['steps_per_host_second']:.1f}",
        "",
        "## Per direction",
        "",
        "| Direction | Pass | RMS error (mm) | Max error (mm) | Mean iterations | Contacts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction, values in document["by_direction"].items():
        lines.append(
            f"| {direction} | {values['pass_count']}/{values['case_count']} | "
            f"{values['position_error_rms_mm']:.6f} | {values['position_error_max_mm']:.6f} | "
            f"{values['mean_iterations']:.3f} | {values['contact_case_count']} |"
        )
    lines.extend(
        [
            "",
            "The JSON artifact contains initial and achieved joint states and poses, requested",
            "and actual deltas, IK status/iterations/residuals, simulation time, and named geom",
            "contact diagnostics for every case.",
            "",
            "## Side effects",
            "",
            "- robot_connections = 0",
            "- robot_commands = 0",
            "- serial_connections = 0",
            "- physical_camera_connections = 0",
            "- openai_requests = 0",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing evidence directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    starts = _load_starts()
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start_name, q_start in starts.items():
        for direction, delta in DIRECTIONS.items():
            cases.append(_run_case(start_name, q_start, direction, delta))
    host_elapsed = time.perf_counter() - started
    by_direction: dict[str, Any] = {}
    for direction in DIRECTIONS:
        selected = [case for case in cases if case["direction"] == direction]
        errors = np.array(
            [case["position_residual_norm_m"] * 1000 for case in selected]
        )
        iterations = np.array(
            [case["ik"]["iterations"] for case in selected], dtype=float
        )
        by_direction[direction] = {
            "case_count": len(selected),
            "pass_count": sum(case["pass"] for case in selected),
            "position_error_rms_mm": float(np.sqrt(np.mean(errors**2))),
            "position_error_max_mm": float(np.max(errors)),
            "mean_iterations": float(np.mean(iterations)),
            "contact_case_count": sum(case["contact_count"] > 0 for case in selected),
        }
    probe = NovaMujocoSimulator(config=SimulatorConfig())
    document = {
        "schema": "nova_mujoco_s1_s2_benchmark_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_revision": _git_revision(),
        "model_path": str(probe.model_path.relative_to(ROOT)),
        "model_sha256": probe.model_sha256,
        "mode": probe.mode,
        "controlled_frame": "link6",
        "action_frame": "base",
        "timestep_s": probe.timestep_s,
        "controller_settings": {
            "type": "bounded linear qpos reference interpolation",
            "classification": "SIMULATION_ENGINEERING_PARAMETERS",
            "max_joint_speed_rad_s": probe.config.max_joint_speed_rad_s,
            "wall_clock_pacing": False,
        },
        "ik_settings": NovaMujocoChunkExecutor(probe).ik.config.__dict__,
        "starts": list(starts),
        "directions": {key: value.tolist() for key, value in DIRECTIONS.items()},
        "cases": cases,
        "summary": _summary(cases, host_elapsed),
        "by_direction": by_direction,
        "performance": {"simulation": _simulation_throughput()},
        "side_effects": {
            "robot_connections": 0,
            "robot_commands": 0,
            "serial_connections": 0,
            "physical_camera_connections": 0,
            "openai_requests": 0,
        },
    }
    (output_dir / "benchmark.json").write_text(json.dumps(document, indent=2) + "\n")
    (output_dir / "summary.md").write_text(_markdown(document))
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    document = run(args.output.resolve())
    print(json.dumps(document["summary"], indent=2))
    print(f"evidence: {args.output.resolve()}")
    return (
        0
        if document["summary"]["pass_count"] == document["summary"]["case_count"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
