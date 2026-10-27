#!/usr/bin/env python3
"""Validate S3 production manipulation with fresh S4 observations, entirely offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark_scene import FROZEN_BASE, ROOT, SCENE_DIR  # noqa: E402
from manipulation_simulator import ManipulationSimulator  # noqa: E402
from nova_cameras import CAMERA_NAMES, NovaMujocoCameraRig, SimObservation  # noqa: E402
from run_s3_benchmark import SIDE_EFFECTS, scripted_episode  # noqa: E402
from run_s4_camera_validation import _montage  # noqa: E402
from s4_offline_guard import install  # noqa: E402

S3_COMMIT = "dea5c927e52162d178a3e97d97d6541771dbcab0"
S4_COMMIT = "269f34681a7fee0da3297df3f6df6f551ccd351e"
PRODUCTION_CAMERA_CONFIG = ROOT / "configs" / "s3_s4_cameras.json"
OBJECT_NAMES = ("blue_block", "black_block", "red_box")
STAGES = (
    "initial_scene",
    "approach",
    "grasp",
    "lifted",
    "transported",
    "lowered",
    "released",
    "final_scene",
)
MONTAGES = {
    "initial_scene": "initial_montage.png",
    "grasp": "grasp_montage.png",
    "lifted": "lifted_montage.png",
    "transported": "transport_montage.png",
    "released": "release_montage.png",
    "final_scene": "final_montage.png",
}
SHAPES = dict(
    zip(CAMERA_NAMES, ((480, 640, 3), (480, 640, 3), (720, 1280, 3)), strict=True)
)


def physics_state(sim: ManipulationSimulator) -> np.ndarray:
    """State sufficient to continue integration, including warm starts/forces."""
    fields = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model, fields))
    mujoco.mj_getState(sim.model, sim.data, state, fields)
    return state


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def production_visibility(rig: NovaMujocoCameraRig, obs: SimObservation) -> dict:
    """Validate actual visible production geom pixels, not fixture-colored patches.

    Segmentation is a diagnostic render and is never part of the RGB observation.
    Ten visible pixels and color thresholds below are engineering test criteria.
    The near wrist view is allowed to exclude the remote distractor/container.
    """
    report = {}
    for camera in CAMERA_NAMES:
        labels, snapshot = rig.diagnostic_geom_ids(camera)
        if snapshot["state_hash"] != obs.extra["nova_mujoco"]["state_hash"]:
            raise RuntimeError("visibility render did not use the observation state")
        entries = {}
        for name in OBJECT_NAMES:
            body = rig.model.body(name).id
            geom_ids = np.flatnonzero(rig.model.geom_bodyid == body)
            mask = np.isin(labels, geom_ids)
            pixels = int(np.count_nonzero(mask))
            color = (
                np.median(obs.images[camera][mask], axis=0) if pixels else np.zeros(3)
            )
            r, g, b = (float(v) for v in color)
            distinct = (
                b > 50 and b > r + 30 and b > g + 30
                if name == "blue_block"
                else r > 50 and r > b + 30 and r > g + 30
                if name == "red_box"
                else max(r, g, b) < 80
            )
            entries[name] = {
                "visible_pixels": pixels,
                "median_rgb": color.tolist(),
                "visible_with_expected_color": bool(pixels >= 10 and distinct),
                "geom_names": [rig.model.geom(int(gid)).name for gid in geom_ids],
            }
        report[camera] = entries
    return report


class CheckpointRecorder:
    """Observe S3's existing phase callbacks without modifying its script or state."""

    def __init__(self, rig: NovaMujocoCameraRig, output: Path | None = None):
        self.rig = rig
        self.output = output
        self.records: list[dict] = []
        self.previous: SimObservation | None = None
        self.initial_fixed: dict = {}
        self.initial_wrist: np.ndarray | None = None
        self.pending_stage: str | None = None

    def __call__(self, stage: str, sim: ManipulationSimulator) -> None:
        self.pending_stage = stage
        before = physics_state(sim)
        acceleration = sim.data.qacc.copy()
        metrics = sim.compute_manipulation_metrics()
        obs = self.rig.observe(
            instruction="Place blue_block inside red_box; leave black_block undisturbed."
        )
        meta = obs.extra["nova_mujoco"]
        visibility = production_visibility(self.rig, obs)
        poses = {name: self.rig.camera_pose(name) for name in CAMERA_NAMES}
        if not self.initial_fixed:
            self.initial_fixed = {name: poses[name] for name in CAMERA_NAMES[:2]}
            self.initial_wrist = poses["wrist_rgb"][0].copy()
        link = sim.model.body("Link6").id
        link_p = sim.data.xpos[link]
        link_r = sim.data.xmat[link].reshape(3, 3)
        wrist = self.rig.specs["wrist_rgb"]
        local_r = np.empty(9)
        mujoco.mju_quat2Mat(local_r, np.array(wrist.quaternion_wxyz))
        wrist_position_error = float(
            np.linalg.norm(poses["wrist_rgb"][0] - (link_p + link_r @ wrist.position_m))
        )
        wrist_rotation_error = float(
            np.linalg.norm(poses["wrist_rgb"][1] - link_r @ local_r.reshape(3, 3))
        )
        fixed_exact = all(
            np.array_equal(poses[name][part], self.initial_fixed[name][part])
            for name in CAMERA_NAMES[:2]
            for part in (0, 1)
        )
        gripper = sim.get_gripper_state()
        expected_eef = np.r_[
            sim.get_tool_pose().position_m, [0, 0, 0], gripper["normalized_gripper"]
        ]
        native_contract = tuple(obs.images) == CAMERA_NAMES and all(
            obs.images[name].shape == SHAPES[name]
            and obs.images[name].dtype == np.uint8
            and obs.images[name].flags.c_contiguous
            and not obs.images[name].flags.writeable
            for name in CAMERA_NAMES
        )
        coherent = obs.state_time == sim.simulation_time_s and all(
            frame["state_time"] == obs.state_time == obs.image_times[name]
            and frame["state_hash"] == meta["state_hash"]
            and frame["frame_generation"] == meta["frame_generation"]
            and frame["render_finished_host_monotonic_s"]
            >= frame["render_started_host_monotonic_s"]
            for name, frame in meta["camera"]["frames"].items()
        )
        previous_meta = (
            None if self.previous is None else self.previous.extra["nova_mujoco"]
        )
        fresh = self.previous is None or (
            meta["frame_generation"] == previous_meta["frame_generation"] + 1
            and meta["state_hash"] != previous_meta["state_hash"]
            and obs.state_time > self.previous.state_time
        )
        checks = {
            "native_rgb_contract": native_contract,
            "coherent_snapshot": coherent,
            "fresh_since_previous_phase": fresh,
            "physics_unchanged": bool(
                np.array_equal(before, physics_state(sim))
                and np.array_equal(acceleration, sim.data.qacc)
                and metrics == sim.compute_manipulation_metrics()
            ),
            "fixed_cameras_unchanged": fixed_exact,
            "wrist_rigid_to_link6": wrist_position_error < 1e-12
            and wrist_rotation_error < 1e-12,
            "eef_state_matches_measured_s3": bool(
                np.allclose(obs.state["eef_state"], expected_eef, rtol=0, atol=1e-6)
                and np.array_equal(obs.state["joint_pos"], sim.get_joint_position())
            ),
            "production_objects_visible": all(
                any(
                    visibility[c][o]["visible_with_expected_color"]
                    for c in CAMERA_NAMES
                )
                for o in OBJECT_NAMES
            ),
            "wrist_sees_blue": visibility["wrist_rgb"]["blue_block"][
                "visible_with_expected_color"
            ],
        }
        record = {
            "stage": stage,
            "checks": checks,
            "state_time": obs.state_time,
            "camera_metadata": meta,
            "state": {k: v.tolist() for k, v in obs.state.items()},
            "gripper": gripper,
            "object_states": {
                name: sim.get_object_state(name) for name in OBJECT_NAMES
            },
            "shapes": {name: list(rgb.shape) for name, rgb in obs.images.items()},
            "dtypes": {name: str(rgb.dtype) for name, rgb in obs.images.items()},
            "image_sha256": {name: _hash(rgb) for name, rgb in obs.images.items()},
            "physics_state_sha256_before": _hash(before),
            "physics_state_sha256_after": _hash(physics_state(sim)),
            "visibility": visibility,
            "wrist_position_error_m": wrist_position_error,
            "wrist_rotation_error": wrist_rotation_error,
            "wrist_displacement_from_reset_m": float(
                np.linalg.norm(poses["wrist_rgb"][0] - self.initial_wrist)
            ),
            "changed_pixel_fraction": {}
            if self.previous is None
            else {
                name: float(np.mean(np.any(rgb != self.previous.images[name], axis=2)))
                for name, rgb in obs.images.items()
            },
        }
        self.records.append(record)
        self.previous = obs
        if self.output is not None:
            for camera, rgb in obs.images.items():
                Image.fromarray(rgb).save(self.output / f"{stage}_{camera}.png")
            if stage in MONTAGES:
                _montage(obs.images).save(self.output / MONTAGES[stage])


def render_failure_probe(rig: NovaMujocoCameraRig) -> dict:
    """Inject one partial-batch error; prove no new observation is returned."""
    from unittest.mock import patch

    before = rig._generation
    state = physics_state(rig.simulator)
    original = rig._draw

    def fail_second(name):
        if name == "right_rgb":
            raise RuntimeError("integration injected right_rgb render failure")
        return original(name)

    caught = None
    with patch.object(rig, "_draw", fail_second):
        try:
            rig.observe()
        except RuntimeError as exc:
            caught = str(exc)
    failed_generation = rig._generation
    recovered = rig.observe()
    return {
        "injected": True,
        "error": caught,
        "generation_before": before,
        "generation_after_failure": failed_generation,
        "recovery_generation": recovered.extra["nova_mujoco"]["frame_generation"],
        "no_observation_returned": caught is not None,
        "physics_unchanged": bool(np.array_equal(state, physics_state(rig.simulator))),
        "pass": bool(
            caught
            and before == failed_generation
            and recovered.extra["nova_mujoco"]["frame_generation"] == before + 1
            and np.array_equal(state, physics_state(rig.simulator))
        ),
    }


def _write_json(path: Path, data) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite integration evidence: {output}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output.mkdir(parents=True, exist_ok=False)
    blocked = install()
    with ManipulationSimulator() as baseline:
        baseline_result = scripted_episode(baseline)
    records, failure_probe, renderer, composition_hash = [], {}, {}, None
    integration_error = None
    result = None
    with ManipulationSimulator() as sim:
        try:
            with NovaMujocoCameraRig(sim, config_path=PRODUCTION_CAMERA_CONFIG) as rig:
                renderer = rig.renderer_info
                composition_hash = hashlib.sha256(rig.scene_xml.encode()).hexdigest()
                recorder = CheckpointRecorder(rig, output)
                records = recorder.records
                result = scripted_episode(sim, snapshot=recorder)
                failure_probe = render_failure_probe(rig)
        except Exception as exc:
            integration_error = {
                "category": "observation_failure",
                "type": type(exc).__name__,
                "reason": str(exc),
            }
        final_metrics = sim.compute_manipulation_metrics()
        final_states = {name: sim.get_object_state(name) for name in OBJECT_NAMES}
        scene_spec = sim.scene_spec.as_dict()
    physics_equal = result is not None and result == baseline_result
    phase_checks = {r["stage"]: all(r["checks"].values()) for r in records}
    success = (
        integration_error is None
        and result is not None
        and result["success"]
        and physics_equal
        and tuple(phase_checks) == STAGES
        and all(phase_checks.values())
        and failure_probe.get("pass", False)
        and not any(blocked.values())
    )
    document = {
        "schema": "nova_mujoco_s3_s4_integration_v1",
        "status": "PASS" if success else "PARTIAL",
        "frozen_inputs": {"s1_s2": FROZEN_BASE, "s3": S3_COMMIT, "s4": S4_COMMIT},
        "production_scene": "benchmark_v0",
        "camera_config": str(PRODUCTION_CAMERA_CONFIG.relative_to(ROOT)),
        "s4_test_fixtures_used": False,
        "integration_error": integration_error,
        "stage_checks": phase_checks,
        "physics_episode_identical_without_rendering": physics_equal,
        "render_failure_probe": failure_probe,
        "manipulation_metrics": final_metrics,
        "manipulation_episode": result,
        "final_object_states": final_states,
        "side_effects": SIDE_EFFECTS.copy(),
        "blocked_io_attempts": blocked.copy(),
    }
    sources = [
        ROOT / name
        for name in (
            "nova2.xml",
            "nova_simulator.py",
            "cartesian_bridge.py",
            "benchmark_scene.py",
            "manipulation_simulator.py",
            "run_s3_benchmark.py",
            "nova_cameras.py",
            "run_s3_s4_integration.py",
            "s4_offline_guard.py",
            "configs/s4_cameras.json",
            "configs/s3_s4_cameras.json",
        )
    ] + sorted(SCENE_DIR.glob("*.xml"))
    _write_json(
        output / "environment.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "render_scene_sha256": composition_hash,
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "renderer": renderer,
            "scene_spec": scene_spec,
            "camera_config": json.loads(PRODUCTION_CAMERA_CONFIG.read_text()),
            "determinism_scope": "same recorded runtime; exact full physics episode with and without rendering",
            "side_effects": SIDE_EFFECTS.copy(),
            "blocked_io_attempts": blocked.copy(),
        },
    )
    _write_json(output / "integration_result.json", document)
    _write_json(
        output / "observation_metadata.json",
        {"schema": "s3_s4_observation_metadata_v1", "observations": records},
    )
    _write_json(
        output / "validation.json",
        {
            "status": document["status"],
            "stage_checks": phase_checks,
            "physics_episode_identical_without_rendering": physics_equal,
            "render_failure_probe_pass": failure_probe.get("pass", False),
            "inside_target": final_metrics["inside_target"],
            "manipulation_success": final_metrics["success"],
            "integration_error": integration_error,
            "quality_gates": "quality_gates.json records the separately executed checks",
            "side_effects": SIDE_EFFECTS.copy(),
            "blocked_io_attempts": blocked.copy(),
        },
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "evidence"
        / "sim_s3_s4"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage_checks": result["stage_checks"],
                "error": result["integration_error"],
                "metrics": result["manipulation_metrics"],
                "evidence": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
