#!/usr/bin/env python3
"""Offline scripted S3 contact grasp/lift/transport/place and evidence runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from benchmark_scene import FROZEN_BASE, ROOT, SCENE_DIR
from manipulation_simulator import (
    ManipulationChunkExecutor,
    ManipulationFault,
    ManipulationSimulator,
)

SIDE_EFFECTS = {
    "robot_connections": 0,
    "robot_commands": 0,
    "serial_connections": 0,
    "physical_camera_connections": 0,
    "openai_requests": 0,
}


class BenchmarkFailure(RuntimeError):
    def __init__(self, category: str, reason: str):
        self.category = category
        super().__init__(reason)


def scripted_episode(
    sim: ManipulationSimulator, *, snapshot: Callable | None = None
) -> dict:
    """Run from explicit reset. Every arm move passes the bounded S2 IK seam.

    All targets derive from scene object specs; none write object coordinates.
    Gripper-only actions use the same seven-value public action boundary.
    """
    sim.reset()
    executor = ManipulationChunkExecutor(sim)
    spec = sim.scene_spec
    records: list[dict] = []
    stages: list[dict] = []
    trajectory: list[dict] = []
    stage = "reset"
    failure = None
    blue_initial = np.array(spec.object("blue_block").initial_position_world_m)
    red_pose = sim.get_object_pose("red_box")
    low, _ = spec.object("red_box").interior_bounds_local_m
    base = sim.data.xpos[sim._base_id].copy()
    # benchmark_v0 is an axis-aligned work surface and container; rotations of
    # a replacement asset must preserve its semantic body frame.
    grasp = blue_initial - base
    lifted = grasp + [0, 0, spec.script.lift_clearance_m]
    over_box = red_pose.position_m - base
    over_box[2] = lifted[2]
    release = red_pose.position_m + [0, 0, low[2]] - base
    release[2] += (
        spec.object("blue_block").dimensions_m[2] / 2
        + spec.script.release_height_above_floor_m
    )

    def sample() -> None:
        trajectory.append(
            {
                "stage": stage,
                "time_s": sim.simulation_time_s,
                "q_rad": sim.get_joint_position().tolist(),
                "grasp_pose_base": sim.get_tool_pose().as_dict(),
                "blue": sim.get_object_state("blue_block"),
                "black": sim.get_object_state("black_block"),
                "gripper": sim.get_gripper_state(),
            }
        )

    def checkpoint(name: str, passed: bool) -> None:
        stages.append(
            {
                "name": name,
                "pass": bool(passed),
                "time_s": sim.simulation_time_s,
                "gripper": sim.get_gripper_state(),
                "metrics": sim.compute_manipulation_metrics(),
            }
        )
        sample()
        if snapshot:
            snapshot(name, sim)

    def actions(waypoints: list) -> None:
        result = executor.execute_chunk(waypoints)
        for record in result.records:
            item = record.as_dict()
            # Zero-motion IK has no computed condition number; JSON must stay strict.
            if not math.isfinite(item["ik"]["condition_number"]):
                item["ik"]["condition_number"] = None
            item["stage"] = stage
            records.append(item)
        if not result.succeeded:
            reason = result.reason or "rejected action"
            category = (
                "IK_fail"
                if "IK_fail" in reason
                else "collision_abort"
                if "collision_abort" in reason
                else "safety/validation_abort"
            )
            raise BenchmarkFailure(category, reason)

    def move(target: np.ndarray, gripper: float, *, require_hold: bool = False) -> None:
        start = sim.get_tool_pose().position_m
        count = max(
            1,
            math.ceil(
                float(np.linalg.norm(target - start)) / spec.physics.waypoint_step_m
            ),
        )
        for position in np.linspace(start, target, count + 1)[1:]:
            actions([[*position, 0, 0, 0, gripper]])
            sample()
            if require_hold and not sim.get_gripper_state()["holding"]:
                raise BenchmarkFailure("dropped", f"lost stable contact during {stage}")

    def gripper_action(value: float, seconds: float) -> None:
        actions([[*sim.get_tool_pose().position_m, 0, 0, 0, value]])
        sim.hold(seconds)
        sample()

    try:
        checkpoint("initial_scene", True)
        stage = "open"
        gripper_action(1, spec.script.initial_settle_s)
        stage = "approach"
        pregrasp = grasp + [0, 0, spec.script.pregrasp_clearance_m]
        move(pregrasp, 1)
        move(grasp, 1)
        checkpoint("approach", True)
        stage = "grasp"
        gripper_action(0, spec.script.gripper_settle_s)
        candidate = sim.get_gripper_state()["state"] == "HOLDING_CANDIDATE"
        checkpoint("grasp", candidate)
        if not candidate:
            raise BenchmarkFailure(
                "grasp_fail", "no stable bilateral contact at grasp pose"
            )
        stage = "lift"
        move(lifted, 0)
        sim.hold(spec.script.lifted_hold_s)
        held = sim.get_gripper_state()["holding"]
        checkpoint("lifted", held)
        if not held:
            raise BenchmarkFailure(
                "grasp_fail", "object did not follow gripper during lift"
            )
        stage = "transport"
        move(over_box, 0, require_hold=True)
        sim.hold(spec.script.transported_hold_s)
        held = sim.get_gripper_state()["holding"]
        checkpoint("transported", held)
        if not held:
            raise BenchmarkFailure("transport_fail", "lost hold over target")
        stage = "lower"
        move(release, 0)
        sim.hold(spec.script.lowered_hold_s)
        checkpoint("lowered", sim.inside_target())
        stage = "release"
        gripper_action(1, spec.script.gripper_settle_s)
        released = sim.compute_manipulation_metrics()["released"]
        checkpoint("released", released)
        if not released:
            raise BenchmarkFailure("release_fail", "object is still contacting fingers")
        stage = "retreat"
        move(over_box, 1)
        sim.hold(spec.script.final_settle_s)
        metrics = sim.compute_manipulation_metrics()
        checkpoint("final_scene", metrics["success"])
        if not metrics["success"]:
            raise BenchmarkFailure(
                "place_miss", "released object failed geometric/settled success checks"
            )
    except BenchmarkFailure as exc:
        failure = {"category": exc.category, "stage": stage, "reason": str(exc)}
    except ManipulationFault as exc:
        failure = {
            "category": "collision_abort"
            if "collision_abort" in str(exc)
            else "safety/validation_abort",
            "stage": stage,
            "reason": str(exc),
        }
    finally:
        executor.close()
    metrics = sim.compute_manipulation_metrics()
    return {
        "schema": "nova_mujoco_s3_benchmark_v1",
        "scene": spec.name,
        "mode": sim.mode,
        "action_frame": "base",
        "controlled_frame": executor.frame,
        "action_order": ["x", "y", "z", "yaw", "pitch", "roll", "gripper"],
        "object_pose_frame": "world",
        "orientation": "reset-relative; zero-only S2 subset",
        "failure": failure,
        "success": bool(metrics["success"] and failure is None),
        "metrics": metrics,
        "stages": stages,
        "waypoints": records,
        "trajectory": trajectory,
        "simulation_time_s": sim.simulation_time_s,
        "side_effects": SIDE_EFFECTS.copy(),
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render_snapshot(path: Path, sim: ManipulationSimulator) -> None:
    """Free inspection view only; no S4 camera names or calibration contract."""
    from PIL import Image

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (-0.16, -0.01, 0.18)
    camera.distance = 1.35
    camera.azimuth = 140
    camera.elevation = -28
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0  # Frozen coarse arm boxes; contact still enabled.
    option.sitegroup[:] = 0
    renderer = mujoco.Renderer(sim.model, height=640, width=640)
    try:
        renderer.update_scene(sim.data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(path)
    finally:
        # MuJoCo 2.3.7 predates Renderer.close/context-manager support. Its GL
        # and MjrContext owners release their resources when references expire.
        if hasattr(renderer, "close"):
            renderer.close()
        del renderer


def run(output_dir: Path, *, render: bool = False) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    render_status: dict = {"requested": render, "images": [], "errors": []}

    def snapshot(name: str, sim: ManipulationSimulator) -> None:
        if not render or name not in {
            "initial_scene",
            "grasp",
            "lifted",
            "transported",
            "released",
            "final_scene",
        }:
            return
        try:
            render_snapshot(output_dir / f"{name}.png", sim)
            render_status["images"].append(f"{name}.png")
        except Exception as exc:
            render_status["errors"].append({"stage": name, "reason": str(exc)})

    with ManipulationSimulator() as sim:
        first = scripted_episode(sim, snapshot=snapshot)
        final_states = {name: sim.get_object_state(name) for name in sim._object_ids}
        contacts = sim.contact_summary()
        timeline = sim.contact_timeline.copy()
        # Independent fresh simulator verifies reproducibility on this runtime.
        with ManipulationSimulator() as other:
            second = scripted_episode(other)
        repeat_equal = first == second
        first["repeat_episode_exact_equal"] = repeat_equal
        weld_count = int(np.count_nonzero(sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
        finger_coupling_count = int(
            np.count_nonzero(sim.model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = [
            ROOT / f
            for f in (
                "benchmark_scene.py",
                "manipulation_simulator.py",
                "run_s3_benchmark.py",
                "nova2.xml",
                "nova_simulator.py",
                "cartesian_bridge.py",
            )
        ]
        tracked.extend(sorted(SCENE_DIR.glob("*.xml")))
        environment = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "base_commit": FROZEN_BASE,
            "source_revision": revision,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tracked
            },
            "scene_spec": sim.scene_spec.as_dict(),
            "timestep_s": sim.timestep_s,
            "render": render_status,
            "side_effects": SIDE_EFFECTS.copy(),
            "determinism_scope": "two fresh instances on the recorded runtime only",
        }
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "final_object_states.json", final_states)
    _write_json(output_dir / "contact_summary.json", contacts)
    _write_json(output_dir / "contact_timeline.json", timeline)
    trajectory = first.pop("trajectory")
    with (output_dir / "trajectory.jsonl").open("w") as stream:
        for row in trajectory:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    _write_json(output_dir / "benchmark_result.json", first)
    _write_json(
        output_dir / "validation.json",
        {
            "status": "PASS"
            if first["success"] and second["success"] and repeat_equal
            else "PARTIAL",
            "first_episode_success": first["success"],
            "repeat_episode_success": second["success"],
            "repeat_entire_episode_exact_equal": repeat_equal,
            "no_object_pose_writes_during_episode": True,
            "object_weld_constraints": weld_count,
            "gripper_coupling_constraints": finger_coupling_count,
            "render": render_status,
            "quality_gate_results": "See quality_gates.json generated separately; runner does not claim tests ran.",
            "side_effects": SIDE_EFFECTS.copy(),
        },
    )
    return first


def viewer() -> None:
    import mujoco.viewer

    with ManipulationSimulator() as sim:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as handle:
            handle.cam.lookat[:] = (-0.16, -0.01, 0.18)
            handle.cam.distance = 1.35
            handle.cam.azimuth = 140
            handle.cam.elevation = -28
            handle.opt.geomgroup[1] = 0
            while handle.is_running():
                sim.step()
                handle.sync()
                time.sleep(sim.timestep_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "evidence"
        / "sim_s3"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="save free-view PNGs (headless: MUJOCO_GL=egl)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="inspect initial scene with a desktop display",
    )
    args = parser.parse_args()
    if args.viewer:
        viewer()
        return 0
    result = run(args.output, render=args.render)
    print(
        json.dumps(
            {
                "success": result["success"],
                "failure": result["failure"],
                "metrics": result["metrics"],
                "evidence": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result["success"] and result["repeat_episode_exact_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
