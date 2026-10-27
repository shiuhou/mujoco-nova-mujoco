#!/usr/bin/env python3
"""Generate new offline S4 evidence, including RGB images and optional frustums.

No GUI, physical devices, network, or providers. Select MUJOCO_GL before this
process starts to override the default EGL backend. Output must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nova_cameras import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CONFIG,
    TEST_FIXTURES,
    NovaMujocoCameraRig,
)
from nova_simulator import NovaMujocoSimulator  # noqa: E402
from s4_offline_guard import install  # noqa: E402
from s4_validation import fixture_report, observation_repeat_report  # noqa: E402

ROOT = Path(__file__).resolve().parent


def _revision() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "source_dirty": bool(git("status", "--porcelain")),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _montage(images: dict[str, np.ndarray]) -> Image.Image:
    montage = Image.new("RGB", (1280, 900), (235, 237, 240))
    draw = ImageDraw.Draw(montage)
    for i, (name, rgb) in enumerate(images.items()):
        pic = Image.fromarray(rgb)
        pic.thumbnail((640, 480) if i < 2 else (640, 360))
        x, y = (i * 640, 30) if i < 2 else (320, 540)
        montage.paste(pic, (x, y))
        draw.text(
            (x + 10, y - 22),
            f"{name} | {rgb.shape[1]} x {rgb.shape[0]} RGB",
            fill=(25, 30, 40),
        )
    return montage


def _timing(rig: NovaMujocoCameraRig, repetitions: int) -> dict:
    results = {}
    for name in (*CAMERA_NAMES, "all_three"):
        render = rig.observe if name == "all_three" else lambda n=name: rig.render(n)
        render()  # Warm context; startup is measured separately.
        started = time.perf_counter()
        for _ in range(repetitions):
            render()
        elapsed = time.perf_counter() - started
        results[name] = {
            "repetitions": repetitions,
            "host_elapsed_s": elapsed,
            "states_per_host_second": repetitions / elapsed,
            "mean_ms_per_state": elapsed * 1000 / repetitions,
            "images_per_state": 3 if name == "all_three" else 1,
        }
    return results


def run(
    output: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    repetitions: int = 20,
    debug: bool = False,
) -> dict:
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    revision = _revision()
    output.mkdir(parents=True, exist_ok=False)
    blocked = install()
    samples = json.loads((ROOT / "real_nova2_samples.json").read_text())["samples"]
    initial_q = np.deg2rad(samples[0]["q_deg"])
    with NovaMujocoSimulator() as sim:
        sim.reset(initial_q)
        started = time.perf_counter()
        with NovaMujocoCameraRig(
            sim, config_path=config_path, fixture_path=TEST_FIXTURES
        ) as rig:
            startup_s = time.perf_counter() - started
            first = rig.observe(
                instruction="S4 visual fixtures only; no manipulation task"
            )
            second = rig.observe()
            same_pixels = {
                n: bool(np.array_equal(first.images[n], second.images[n]))
                for n in CAMERA_NAMES
            }
            static_repeat = observation_repeat_report(first, second)
            fixtures = fixture_report(rig, first)
            for name in CAMERA_NAMES:
                Image.fromarray(first.images[name]).save(output / f"{name}.png")
            _montage(first.images).save(output / "montage.png")
            if debug:
                Image.fromarray(rig.render_debug()).save(output / "camera_debug.png")
            initial_poses = {n: rig.camera_pose(n) for n in CAMERA_NAMES}
            movement_records = []
            for i, delta in enumerate(
                (
                    np.array([0.12, -0.04, 0.03, 0, 0, 0.08]),
                    np.array([-0.10, 0.04, -0.02, 0.03, 0, -0.06]),
                )
            ):
                sim.reset(initial_q)
                sim.command_joint_target(initial_q + delta)
                sim.settle()
                observation = rig.observe()
                poses = {n: rig.camera_pose(n) for n in CAMERA_NAMES}
                parent = sim.get_link6_pose()
                record = {
                    "label": f"scripted_joint_state_{i + 1}",
                    "q_rad": sim.get_joint_position().tolist(),
                    "state_time": observation.state_time,
                    "state": {k: v.tolist() for k, v in observation.state.items()},
                    "camera_metadata": observation.extra["nova_mujoco"],
                    "fixed_camera_position_changes_m": {
                        n: float(np.linalg.norm(poses[n][0] - initial_poses[n][0]))
                        for n in CAMERA_NAMES[:2]
                    },
                    "fixed_camera_rotation_changes": {
                        n: float(np.linalg.norm(poses[n][1] - initial_poses[n][1]))
                        for n in CAMERA_NAMES[:2]
                    },
                    "wrist_position_change_m": float(
                        np.linalg.norm(
                            poses["wrist_rgb"][0] - initial_poses["wrist_rgb"][0]
                        )
                    ),
                    "link6_pose_base": parent.as_dict(),
                    "changed_pixel_fraction": {
                        n: float(
                            np.mean(
                                np.any(first.images[n] != observation.images[n], axis=2)
                            )
                        )
                        for n in CAMERA_NAMES
                    },
                    "fixture_visibility": fixture_report(rig, observation),
                }
                movement_records.append(record)
                _montage(observation.images).save(output / f"movement_{i + 1}.png")
            sim.reset(initial_q)
            reset_obs = rig.observe()
            reset_same_pixels = {
                n: bool(np.array_equal(first.images[n], reset_obs.images[n]))
                for n in CAMERA_NAMES
            }
            reset_repeat = observation_repeat_report(first, reset_obs)
            performance = _timing(rig, repetitions)
            roles = {
                "schema": "nova_mujoco_camera_roles_v1",
                "calibration": "No measured intrinsics, distortion, extrinsics or hand-eye transform",
                "roles": [
                    {
                        "logical_name": entry["parameters"]["name"],
                        "physical_identity": entry["physical_identity"],
                        "role": entry["parameters"]["role"],
                        "parent": entry["parameters"]["parent"],
                        "resolution_wh": [
                            entry["parameters"]["width"],
                            entry["parameters"]["height"],
                        ],
                        "provenance": entry["provenance"],
                    }
                    for entry in rig.config_document["cameras"]
                ],
            }
            success = (
                static_repeat["passed"]
                and reset_repeat["passed"]
                and all(
                    f["color_visible"]
                    for cam in fixtures.values()
                    for f in cam.values()
                )
                and all(
                    all(v == 0 for v in r["fixed_camera_position_changes_m"].values())
                    and all(v == 0 for v in r["fixed_camera_rotation_changes"].values())
                    and r["wrist_position_change_m"] > 0.001
                    and all(v > 0.0001 for v in r["changed_pixel_fraction"].values())
                    for r in movement_records
                )
                and all(v == 0 for v in blocked.values())
            )
            document = {
                "schema": "nova_mujoco_s4_validation_v1",
                "status": "PASS" if success else "FAIL",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git": revision,
                "frozen_base": "88d93cf686ba0c67afe06b163aa7fb48d65f1a07",
                "files_sha256": {
                    p.name: _hash(p)
                    for p in (
                        ROOT / "nova2.xml",
                        config_path,
                        TEST_FIXTURES,
                        ROOT / "nova_cameras.py",
                        ROOT / "run_s4_camera_validation.py",
                        ROOT / "s4_validation.py",
                    )
                },
                "render_scene_sha256": hashlib.sha256(
                    rig.scene_xml.encode()
                ).hexdigest(),
                "renderer": rig.renderer_info,
                "startup_host_s": startup_s,
                "initial_sample": samples[0]["name"],
                "q_start_rad": initial_q.tolist(),
                "state": {k: v.tolist() for k, v in first.state.items()},
                "state_time": first.state_time,
                "camera_metadata": first.extra["nova_mujoco"],
                "shapes": {n: list(rgb.shape) for n, rgb in first.images.items()},
                "dtypes": {n: str(rgb.dtype) for n, rgb in first.images.items()},
                "static_repeat_exact": same_pixels,
                "reset_repeat_exact": reset_same_pixels,
                "static_repeat_validation": static_repeat,
                "reset_repeat_validation": reset_repeat,
                "image_sha256": {
                    n: hashlib.sha256(rgb.tobytes()).hexdigest()
                    for n, rgb in first.images.items()
                },
                "fixture_visibility": fixtures,
                "movement_checks": movement_records,
                "performance": performance,
                "blocked_io_attempts": blocked,
                "side_effects": {
                    "robot_connections": 0,
                    "robot_commands": 0,
                    "serial_connections": 0,
                    "physical_camera_connections": 0,
                    "openai_requests": 0,
                },
                "limitations": [
                    "Engineering camera transforms/FOV; not calibrated",
                    "Test-only visual fixtures; no S3 physics",
                    "Same-state render coherence is not physical exposure synchronization",
                    "Exact pixel identity reported separately; S4R permits at most eight 1-LSB pixels with exact state/camera parity and fresh generation",
                ],
            }
            _save_json(output / "camera_config.json", rig.config_document)
            _save_json(output / "camera_roles.json", roles)
            _save_json(output / "validation.json", document)
            lines = [
                "# S4 three-camera observation evidence",
                "",
                f"Status: **{document['status']}**",
                "",
                f"Code revision: `{revision['head']}`",
                "",
                "Native-resolution uint8 RGB; test fixtures only. No physical calibration claim.",
                "",
                "| View | States/s (FPS for one view) | Mean ms/state |",
                "|---|---:|---:|",
            ]
            lines.extend(
                f"| {n} | {v['states_per_host_second']:.2f} | {v['mean_ms_per_state']:.2f} |"
                for n, v in performance.items()
            )
            lines.extend(
                [
                    "",
                    "All camera images use one copied simulation state; static/reset state and camera parity are exact. Pixel identity and bounded S4R residuals are reported separately.",
                    "Fixed cameras remain fixed; wrist follows Link6. Parameters and full residuals are in JSON.",
                    "",
                    "No robot, serial, physical camera, or OpenAI access.",
                ]
            )
            (output / "summary.md").write_text("\n".join(lines) + "\n")
            return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "evidence"
        / "sim_s4"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--debug", action="store_true", help="save offline camera axes/frustums image"
    )
    args = parser.parse_args()
    report = run(
        args.output,
        config_path=args.config,
        repetitions=args.repetitions,
        debug=args.debug,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "performance": report["performance"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
