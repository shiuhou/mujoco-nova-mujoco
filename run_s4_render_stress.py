#!/usr/bin/env python3
"""Local S4 render reproducibility diagnostics; no provider or hardware access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from nova_cameras import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CONFIG,
    TEST_FIXTURES,
    NovaMujocoCameraRig,
)
from nova_simulator import NovaMujocoSimulator  # noqa: E402
from s4_offline_guard import install  # noqa: E402
from s4_validation import observation_repeat_report, rgb_repeat_report  # noqa: E402

ROOT = Path(__file__).resolve().parent


class DiagnosticRig(NovaMujocoCameraRig):
    """Explicit experiment variants, never selected by production observation code."""

    def __init__(self, sim, output, intervention):
        self.intervention = intervention
        self.readback_checks = []
        self.readback_count = 0
        self.scene_hashes = {}
        self.scene_references = {}
        self.scene_differences = {}
        self.frozen_scenes = {}
        self.output = output
        config = DEFAULT_CONFIG
        if intervention == "msaa0":
            config = output / "experimental_config.json"
        super().__init__(sim, fixture_path=TEST_FIXTURES, config_path=config)
        from OpenGL import GL
        from OpenGL import EGL

        self.renderer_info["gl_dither_enabled"] = bool(GL.glIsEnabled(GL.GL_DITHER))
        self.renderer_info["intervention"] = intervention
        if os.environ["MUJOCO_GL"] == "egl":
            display = EGL.eglGetCurrentDisplay()
            self.renderer_info["egl"] = {
                "vendor": EGL.eglQueryString(display, EGL.EGL_VENDOR).decode(),
                "version": EGL.eglQueryString(display, EGL.EGL_VERSION).decode(),
                "client_apis": EGL.eglQueryString(
                    display, EGL.EGL_CLIENT_APIS
                ).decode(),
            }

    def _prepare_view(self, name):
        if self.intervention == "frozen_scene" and name in self.frozen_scenes:
            self._scene = self.frozen_scenes[name]
            self._gl.make_current()
            spec = self.specs[name]
            return mujoco.MjrRect(0, 0, spec.width, spec.height)
        if self.intervention == "frozen_scene":
            self._scene = mujoco.MjvScene(
                self.model, maxgeom=self.renderer_config.max_geoms
            )
        rect = super()._prepare_view(name)

        def fields(obj, excluded=()):
            result = {}
            for key in dir(obj):
                if key.startswith("_") or key in excluded:
                    continue
                value = getattr(obj, key)
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool)):
                    result[key] = value
            return result

        # mjvisualize.h marks camdist/transparent as internally computed rendering
        # fields; exclude these and geomorder, not actual camera/geometry inputs.
        inputs = {
            "scene": fields(self._scene, ("geomorder",)),
            "geoms": [
                fields(g, ("camdist", "transparent"))
                for g in self._scene.geoms[: self._scene.ngeom]
            ],
            "lights": [
                fields(light) for light in self._scene.lights[: self._scene.nlight]
            ],
            "camera": [fields(c) for c in self._scene.camera],
        }
        if name not in self.scene_references:
            self.scene_references[name] = inputs

        def differences(a, b, path=""):
            if isinstance(a, dict):
                return [
                    item
                    for key in a
                    for item in differences(a[key], b[key], f"{path}/{key}")
                ]
            if isinstance(a, list):
                if len(a) != len(b):
                    return [{"path": path, "before": a, "after": b}]
                return [
                    item
                    for i, (x, y) in enumerate(zip(a, b, strict=True))
                    for item in differences(x, y, f"{path}/{i}")
                ]
            return [] if a == b else [{"path": path, "before": a, "after": b}]

        self.scene_differences[name] = differences(self.scene_references[name], inputs)
        self.scene_hashes[name] = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        if self.intervention == "frozen_scene":
            self.frozen_scenes[name] = self._scene
        return rect

    def _draw(self, name):
        from OpenGL import GL

        if self.intervention not in ("finish", "no_dither"):
            rgb = super()._draw(name)
        else:
            rect = self._prepare_view(name)
            if self.intervention == "no_dither":
                GL.glDisable(GL.GL_DITHER)
            if self.intervention == "finish":
                GL.glFinish()
            rgb = np.empty((rect.height, rect.width, 3), dtype=np.uint8)
            mujoco.mjr_render(rect, self._scene, self._context)
            if self.intervention == "finish":
                GL.glFinish()
            mujoco.mjr_readPixels(rgb, None, rect, self._context)
            rgb = np.flipud(rgb).copy()
            rgb.flags.writeable = False
        if self.intervention == "readback":
            spec = self.specs[name]
            rect = mujoco.MjrRect(0, 0, spec.width, spec.height)
            again = np.zeros_like(rgb)
            mujoco.mjr_readPixels(again, None, rect, self._context)
            again = np.flipud(again).copy()
            diff = pixel_difference(rgb, again)
            if diff["differing_pixels"]:
                folder = self.output / f"readback-{self.readback_count:05d}-{name}"
                folder.mkdir()
                Image.fromarray(rgb).save(folder / "reference.png")
                Image.fromarray(again).save(folder / "mismatch.png")
                delta = np.abs(rgb.astype(np.int16) - again.astype(np.int16)).astype(
                    np.uint8
                )
                Image.fromarray(delta).save(folder / "absolute_difference.png")
                (folder / "difference.json").write_text(
                    json.dumps(diff, indent=2) + "\n"
                )
            self.readback_checks.append({"camera": name, **diff})
            self.readback_count += 1
        return rgb


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def pixel_difference(reference: np.ndarray, image: np.ndarray) -> dict:
    delta = np.abs(reference.astype(np.int16) - image.astype(np.int16))
    changed = np.any(delta != 0, axis=2)
    y, x = np.nonzero(changed)
    return {
        "differing_pixels": int(changed.sum()),
        "differing_channels": int(np.count_nonzero(delta)),
        "max_channel_difference": int(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "bbox_xyxy_inclusive": (
            [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None
        ),
    }


def record(rig: NovaMujocoCameraRig, observation) -> dict:
    meta = observation.extra["nova_mujoco"]
    return {
        "simulator_state_hash": meta["state_hash"],
        "qpos": rig.simulator.data.qpos.tolist(),
        "qvel": rig.simulator.data.qvel.tolist(),
        "sim_time": observation.state_time,
        "generation": meta["frame_generation"],
        "camera": {
            name: {
                key: value
                for key, value in meta["camera"]["frames"][name].items()
                if key in ("camera_world_position_m", "camera_to_world_rotation")
            }
            for name in CAMERA_NAMES
        },
        "derived_state_hashes": {
            name: digest(getattr(rig.data, name))
            for name in (
                "xpos",
                "xmat",
                "geom_xpos",
                "geom_xmat",
                "cam_xpos",
                "cam_xmat",
            )
        },
        "image_hashes": {name: digest(rgb) for name, rgb in observation.images.items()},
        "renderer": rig.renderer_info,
        "scene_hashes": dict(getattr(rig, "scene_hashes", {})),
        "scene_differences_from_first_view": dict(
            getattr(rig, "scene_differences", {})
        ),
    }


def run(
    output: Path, cycles: int, modes: list[str], intervention: str = "none"
) -> dict:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    output.mkdir(parents=True, exist_ok=False)
    blocked = install()
    samples = json.loads((ROOT / "real_nova2_samples.json").read_text())["samples"]
    q = np.deg2rad(samples[0]["q_deg"])
    if intervention == "msaa0":
        config = json.loads(DEFAULT_CONFIG.read_text())
        config["renderer"]["parameters"]["multisamples"] = 0
        (output / "experimental_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
    summary = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "cycles_per_mode": cycles,
        "intervention": intervention,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "nova_cameras.py",
                "nova_simulator.py",
                "run_s4_render_stress.py",
            )
        },
        "modes": {},
        "blocked_io_attempts": blocked,
        "side_effects": dict.fromkeys(
            (
                "provider_requests",
                "robot_connections",
                "robot_commands",
                "serial_connections",
                "physical_camera_connections",
            ),
            0,
        ),
    }
    for mode in modes:
        counts = dict.fromkeys(CAMERA_NAMES, 0)
        validation_failures = 0
        mismatches = []
        with NovaMujocoSimulator() as sim:
            sim.reset(q)
            rig = DiagnosticRig(sim, output, intervention)
            try:
                reference = rig.observe()
                ref_record = record(rig, reference)
                for name, rgb in reference.images.items():
                    Image.fromarray(rgb).save(output / f"{mode}-reference-{name}.png")
                with (output / f"{mode}.jsonl").open("x") as stream:
                    stream.write(json.dumps({"iteration": -1, **ref_record}) + "\n")
                    for i in range(cycles):
                        if mode == "fresh":
                            rig.close()
                            rig = DiagnosticRig(sim, output, intervention)
                        if mode == "history_reset":
                            rig.render_debug()
                            for delta in (
                                [0.12, -0.04, 0.03, 0, 0, 0.08],
                                [-0.10, 0.04, -0.02, 0.03, 0, -0.06],
                            ):
                                sim.reset(q)
                                sim.command_joint_target(q + delta)
                                sim.settle()
                                rig.observe()
                        if mode in ("reset", "history_reset"):
                            sim.reset(q)
                        observation = rig.observe()
                        row = record(rig, observation)
                        if mode == "fresh":
                            # Generations are local to a rig, so compare independent
                            # instances' state/camera data, not their generation IDs.
                            validation = {
                                "passed": all(
                                    row[key] == ref_record[key]
                                    for key in (
                                        "simulator_state_hash",
                                        "qpos",
                                        "qvel",
                                        "sim_time",
                                        "camera",
                                        "derived_state_hashes",
                                        "scene_hashes",
                                    )
                                )
                                and all(
                                    rgb_repeat_report(
                                        reference.images[n], observation.images[n]
                                    )["passed"]
                                    for n in CAMERA_NAMES
                                )
                            }
                        else:
                            validation = observation_repeat_report(
                                reference, observation
                            )
                        row["repeat_validation"] = validation
                        row["rig_instance"] = i + 1 if mode == "fresh" else 0
                        validation_failures += not validation["passed"]
                        stream.write(json.dumps({"iteration": i, **row}) + "\n")
                        for name in CAMERA_NAMES:
                            rgb = observation.images[name]
                            if (
                                row["image_hashes"][name]
                                == ref_record["image_hashes"][name]
                            ):
                                continue
                            counts[name] += 1
                            folder = output / f"{mode}-{i:04d}-{name}"
                            folder.mkdir()
                            ref_rgb = reference.images[name]
                            Image.fromarray(ref_rgb).save(folder / "reference.png")
                            Image.fromarray(rgb).save(folder / "mismatch.png")
                            delta = np.abs(
                                ref_rgb.astype(np.int16) - rgb.astype(np.int16)
                            ).astype(np.uint8)
                            Image.fromarray(delta).save(
                                folder / "absolute_difference.png"
                            )
                            stats = {
                                "iteration": i,
                                "camera_name": name,
                                **pixel_difference(ref_rgb, rgb),
                                "same_state": row["simulator_state_hash"]
                                == ref_record["simulator_state_hash"],
                                "same_camera": row["camera"] == ref_record["camera"],
                                "same_derived_state": row["derived_state_hashes"]
                                == ref_record["derived_state_hashes"],
                                "same_scene": row["scene_hashes"]
                                == ref_record["scene_hashes"],
                            }
                            (folder / "difference.json").write_text(
                                json.dumps(stats, indent=2) + "\n"
                            )
                            mismatches.append(stats)
                summary["modes"][mode] = {
                    "mismatch_counts": counts,
                    "mismatches": mismatches,
                    "readback_checks": rig.readback_checks,
                    "repeat_validation_failures": validation_failures,
                }
                print(f"{mode}: {counts}", flush=True)
            finally:
                rig.close()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument(
        "--intervention",
        choices=("none", "readback", "finish", "no_dither", "msaa0", "frozen_scene"),
        default="none",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("static", "reset", "fresh", "history_reset"),
        default=["static", "reset", "fresh", "history_reset"],
    )
    args = parser.parse_args()
    run(args.output, args.cycles, args.modes, args.intervention)
