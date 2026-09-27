"""Offline RGB observation boundary; no devices, networking, or agent dependency.

Set MUJOCO_GL=egl (or osmesa) before importing MuJoCo. Camera parameters are
explicit assumptions unless provenance says VERIFIED_FROM_EXISTING_EVIDENCE.
The source simulator/model is never changed; a render-only copy snapshots its
entire generalized state, including future object joints and mocap bodies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from nova_simulator import NovaMujocoSimulator

ROOT = Path(__file__).resolve().parent
CAMERA_NAMES = ("front_rgb", "right_rgb", "wrist_rgb")
DEFAULT_CONFIG = ROOT / "configs" / "s4_cameras.json"
TEST_FIXTURES = ROOT / "fixtures" / "s4" / "visual_fixtures.xml"
VERIFIED = "VERIFIED_FROM_EXISTING_EVIDENCE"
ENGINEERING = "SIMULATION_ENGINEERING_PARAMETER"


def _vector(values: Any, size: int, label: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class CameraSpec:
    name: str
    role: str
    parent: str
    position_m: tuple[float, ...]
    quaternion_wxyz: tuple[float, ...]
    width: int
    height: int
    fovy_deg: float

    def __post_init__(self) -> None:
        if self.name not in CAMERA_NAMES:
            raise ValueError(f"unknown camera name {self.name!r}")
        expected_parent = "Link6" if self.name == "wrist_rgb" else "world"
        if self.parent != expected_parent:
            raise ValueError(f"{self.name} requires parent {expected_parent}")
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, "position"))
        quat = _vector(self.quaternion_wxyz, 4, "quaternion")
        if not math.isclose(float(np.linalg.norm(quat)), 1.0, abs_tol=1e-8):
            raise ValueError(
                "camera quaternion must be unit length; no silent normalization"
            )
        object.__setattr__(self, "quaternion_wxyz", quat)
        for value in (self.width, self.height):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 4096
            ):
                raise ValueError("camera dimensions must be integers in [1, 4096]")
        if not math.isfinite(self.fovy_deg) or not 1 < self.fovy_deg < 179:
            raise ValueError("fovy_deg must be finite and inside (1, 179)")


@dataclass(frozen=True)
class RendererConfig:
    near_m: float = 0.01
    far_m: float = 10.0
    multisamples: int = 4
    max_geoms: int = 10000

    def __post_init__(self) -> None:
        if not (math.isfinite(self.near_m) and math.isfinite(self.far_m)):
            raise ValueError("clipping distances must be finite")
        if not 0 < self.near_m < self.far_m:
            raise ValueError("require 0 < near_m < far_m")
        if type(self.multisamples) is not int or self.multisamples not in (0, 2, 4, 8):
            raise ValueError("multisamples must be one of 0, 2, 4, 8")
        if type(self.max_geoms) is not int or self.max_geoms < 100:
            raise ValueError("max_geoms must be an integer >= 100")


@dataclass(frozen=True, eq=False)
class CameraFrame:
    rgb: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, eq=False)
class SimObservation:
    """Direct keyword-field mapping to inspect_robots.types.Observation in S5."""

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    image_times: dict[str, float]
    state_time: float
    extra: dict[str, Any]
    instruction: str | None = None


def load_camera_config(
    path: Path = DEFAULT_CONFIG,
) -> tuple[tuple[CameraSpec, ...], RendererConfig]:
    document = json.loads(Path(path).read_text())
    if document["schema"] != "nova_mujoco_camera_rig_v1":
        raise ValueError("unsupported camera config schema")
    specs = tuple(CameraSpec(**entry["parameters"]) for entry in document["cameras"])
    if len(specs) != 3 or {spec.name for spec in specs} != set(CAMERA_NAMES):
        raise ValueError("rig must contain exactly front_rgb, right_rgb, wrist_rgb")
    for entry in document["cameras"]:
        for field in entry["parameters"]:
            source = entry["provenance"].get(field)
            if (
                not source
                or source["status"] not in (VERIFIED, ENGINEERING)
                or not source["source"]
            ):
                raise ValueError(f"missing/invalid provenance for {field}")
    renderer_source = document["renderer"]
    if renderer_source.get("status") != ENGINEERING or not renderer_source.get(
        "source"
    ):
        raise ValueError("renderer settings require engineering provenance")
    return specs, RendererConfig(**document["renderer"]["parameters"])


def _child(parent: ET.Element, name: str) -> ET.Element:
    result = parent.find(name)
    return result if result is not None else ET.SubElement(parent, name)


def _numbers(values: Any) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _expanded_scene(source: Path) -> ET.Element:
    """Expand local MJCF includes before adding cameras; never write source files.

    Like MuJoCo, resolve include paths relative to the main MJCF directory and
    reject duplicate inclusion (including cycles). The included wrapper itself
    is discarded; its children take the include's exact position in the tree.
    """
    source = source.resolve()
    root = ET.parse(source).getroot()
    seen = {source}

    def expand(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag != "include":
                expand(child)
                continue
            filename = child.get("file")
            if not filename:
                raise ValueError("MJCF include requires a file")
            path = (source.parent / filename).resolve()
            if path in seen:
                raise ValueError(f"duplicate or cyclic MJCF include: {filename}")
            seen.add(path)
            included = ET.parse(path).getroot()
            expand(included)
            index = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(included):
                parent.insert(index + offset, element)

    expand(root)
    return root


def camera_scene_xml(
    source: Path,
    specs: tuple[CameraSpec, ...],
    renderer: RendererConfig,
    fixture_path: Path | None = None,
) -> str:
    """Compose cameras into a standalone MJCF without editing source geometry.

    Keep added scene objects in the caller's source MJCF. The optional S4 fixture
    file may contain only static world geoms; it adds no object physics or DOFs.
    Local MJCF includes are expanded in memory, preserving source scene ownership.
    """
    source = Path(source).resolve()
    root = _expanded_scene(source)
    compiler = _child(root, "compiler")
    for key in ("meshdir", "texturedir", "assetdir"):
        if key in compiler.attrib:
            compiler.set(key, str((source.parent / compiler.get(key, "")).resolve()))
    for asset in root.findall("./asset/*[@file]"):
        if asset.tag in ("mesh", "texture"):
            directory = compiler.get(
                asset.tag + "dir", compiler.get("assetdir", str(source.parent))
            )
        else:
            directory = str(source.parent)
        asset.set("file", str((Path(directory) / asset.attrib["file"]).resolve()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("source model has no worldbody")
    for spec in specs:
        if root.find(f".//camera[@name='{spec.name}']") is not None:
            raise ValueError(f"camera name already exists: {spec.name}")
        parent = (
            world
            if spec.parent == "world"
            else world.find(f".//body[@name='{spec.parent}']")
        )
        if parent is None:
            raise ValueError(f"missing camera parent {spec.parent}")
        ET.SubElement(
            parent,
            "camera",
            name=spec.name,
            mode="fixed",
            pos=_numbers(spec.position_m),
            quat=_numbers(spec.quaternion_wxyz),
            fovy=str(spec.fovy_deg),
            ipd="0",
        )
    if fixture_path is not None:
        for geom in ET.parse(fixture_path).getroot():
            if (
                geom.tag != "geom"
                or geom.get("contype") != "0"
                or geom.get("conaffinity") != "0"
            ):
                raise ValueError("S4 visual fixtures must be non-contact static geoms")
            world.append(geom)
    visual = _child(root, "visual")
    _child(visual, "global").set("offwidth", str(max(spec.width for spec in specs)))
    _child(visual, "global").set("offheight", str(max(spec.height for spec in specs)))
    _child(visual, "quality").set("offsamples", str(renderer.multisamples))
    return ET.tostring(root, encoding="unicode")


def relative_orientation(rotation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Inverse of Rz(yaw) Ry(-pitch) Rx(roll) R_reference; all radians."""
    relative = rotation @ reference.T
    pitch = math.asin(float(np.clip(relative[2, 0], -1, 1)))
    if abs(pitch) >= math.pi / 2 - 1e-6:
        raise ValueError("relative orientation inside Euler singularity exclusion")
    angles = np.array(
        [
            math.atan2(relative[1, 0], relative[0, 0]),
            pitch,
            math.atan2(relative[2, 1], relative[2, 2]),
        ]
    )
    angles[np.abs(angles) < 1e-14] = 0
    return angles


class NovaMujocoCameraRig:
    """Coherent snapshots with one offscreen context; single-threaded ownership.

    Calls never step/reset the simulator. Do not mutate the source concurrently
    with rendering. Model configuration is static for the lifetime of a rig;
    rebuild the rig after changing source model geometry/materials.
    """

    def __init__(
        self,
        simulator: NovaMujocoSimulator,
        *,
        config_path: Path = DEFAULT_CONFIG,
        fixture_path: Path | None = None,
        controlled_frame: str | None = None,
    ) -> None:
        if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
            raise RuntimeError(
                "set MUJOCO_GL=egl or osmesa before importing MuJoCo; no GUI fallback"
            )
        self.simulator = simulator
        self.controlled_frame = controlled_frame or simulator.config.default_frame
        simulator.get_tool_pose(
            self.controlled_frame
        )  # Validate before allocating GL resources.
        specs, self.renderer_config = load_camera_config(config_path)
        self.specs = {
            name: next(s for s in specs if s.name == name) for name in CAMERA_NAMES
        }
        self.config_document = json.loads(Path(config_path).read_text())
        self.scene_xml = camera_scene_xml(
            simulator.model_path,
            tuple(self.specs.values()),
            self.renderer_config,
            fixture_path,
        )
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        for field in ("nq", "nv", "na", "nmocap", "nuserdata"):
            if getattr(self.model, field) != getattr(simulator.model, field):
                raise ValueError(
                    "render scene must preserve source generalized-state layout"
                )
        self.model.opt.timestep = simulator.timestep_s
        self.model.vis.map.znear = self.renderer_config.near_m / self.model.stat.extent
        self.model.vis.map.zfar = self.renderer_config.far_m / self.model.stat.extent
        self.data = mujoco.MjData(self.model)
        self._ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in CAMERA_NAMES
        }
        self._scene = mujoco.MjvScene(
            self.model, maxgeom=self.renderer_config.max_geoms
        )
        self._options = mujoco.MjvOption()
        self._options.geomgroup[1] = 0  # Existing coarse collision proxies.
        self._options.geomgroup[3:] = 1  # Added scene objects and S4 visual fixtures.
        self._options.sitegroup[:] = 0  # No diagnostic TCP markers in model-facing RGB.
        self._context = None
        self._gl = None
        self._closed = False
        self._generation = 0
        try:
            self._gl = mujoco.GLContext(
                self.model.vis.global_.offwidth, self.model.vis.global_.offheight
            )
            self._gl.make_current()
            self._context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_100
            )
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
            # Context is current; metadata only, no connection/device capture API.
            from OpenGL import GL

            self.renderer_info = {
                "mujoco_version": mujoco.__version__,
                "backend": os.environ["MUJOCO_GL"],
                "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                "opengl_version": GL.glGetString(GL.GL_VERSION).decode(),
                "parameters": asdict(self.renderer_config),
                "resizing": False,
                "color_order": "RGB",
                "depth": False,
            }
            self._snapshot()
        except BaseException:
            self.close()
            raise

    def _snapshot(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("camera rig is closed")
        self.simulator.get_joint_position()  # Fail if source has been closed.
        source = self.simulator.data
        digest = hashlib.sha256()
        for field in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "userdata",
        ):
            values = getattr(source, field).copy()
            getattr(self.data, field)[:] = values
            digest.update(values.tobytes())
        self.data.time = float(source.time)
        digest.update(np.float64(self.data.time).tobytes())
        mujoco.mj_forward(self.model, self.data)
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        site_name = (
            "tcp" if self.controlled_frame == "link6" else "nominal_grasp_center"
        )
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        base_r = self.data.xmat[base_id].reshape(3, 3)
        xyz = base_r.T @ (self.data.site_xpos[site_id] - self.data.xpos[base_id])
        rotation = base_r.T @ self.data.site_xmat[site_id].reshape(3, 3)
        reference = self.simulator.reference_rotation
        # S1/S2 have no active gripper. S3 exposes measured aperture through its
        # state API; use it without importing or redefining manipulation logic.
        gripper_state = getattr(self.simulator, "get_gripper_state", None)
        gripper = (
            0.0
            if gripper_state is None
            else float(gripper_state()["normalized_gripper"])
        )
        if not math.isfinite(gripper):
            raise ValueError("nonfinite measured gripper state")
        eef = np.r_[xyz, relative_orientation(rotation, reference), gripper]
        q = np.array(
            [
                self.data.qpos[
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                        )
                    ]
                ]
                for i in range(1, 7)
            ]
        )
        return {
            "state_time": float(self.data.time),
            "state_hash": digest.hexdigest(),
            "state": {"eef_state": eef, "joint_pos": q},
        }

    def camera_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and camera-to-world rotation of current source state."""
        self._snapshot()
        return self._pose(name)

    def _pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        camera_id = self._ids[name]
        return (
            self.data.cam_xpos[camera_id].copy(),
            self.data.cam_xmat[camera_id].reshape(3, 3).copy(),
        )

    def project(self, name: str, points_world: Any) -> tuple[np.ndarray, np.ndarray]:
        """Pinhole pixel coordinates and positive view depth; no occlusion claim.

        MuJoCo camera axes: +X right, +Y up, optical axis -Z. Pixel origin is
        top-left; square pixels, vertical FOV, no distortion.
        """
        self._snapshot()
        position, rotation = self._pose(name)
        points = np.asarray(points_world, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_world must be finite (N,3)")
        local = (points - position) @ rotation
        depth = -local[:, 2]
        spec = self.specs[name]
        focal = spec.height / (2 * math.tan(math.radians(spec.fovy_deg) / 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.column_stack(
                (
                    spec.width / 2 + focal * local[:, 0] / depth,
                    spec.height / 2 - focal * local[:, 1] / depth,
                )
            )
        return uv, depth

    def _prepare_view(self, name: str) -> mujoco.MjrRect:
        spec = self.specs[name]
        self._gl.make_current()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self._ids[name]
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self._options,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        return mujoco.MjrRect(0, 0, spec.width, spec.height)

    def _draw(self, name: str) -> np.ndarray:
        spec = self.specs[name]
        rect = self._prepare_view(name)
        rgb = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
        mujoco.mjr_render(rect, self._scene, self._context)
        mujoco.mjr_readPixels(rgb, None, rect, self._context)
        rgb = np.flipud(rgb).copy()  # OpenGL bottom-left -> public top-left.
        rgb.flags.writeable = False
        return rgb

    def diagnostic_geom_ids(self, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Occlusion-aware geometry IDs for validation only, outside RGB observations.

        Uses MuJoCo's native segmentation render, with the same view, clipping and
        visible geom groups as RGB. Background/non-geom pixels are -1. It does
        not commit an observation generation. Render flags are always restored.
        """
        if name not in self.specs:
            raise ValueError(f"unknown camera {name!r}")
        snapshot = self._snapshot()
        rect = self._prepare_view(name)
        flags = self._scene.flags.copy()
        try:
            self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
            pixels = np.empty((rect.height, rect.width, 3), dtype=np.uint8)
            mujoco.mjr_render(rect, self._scene, self._context)
            mujoco.mjr_readPixels(pixels, None, rect, self._context)
            channels = pixels.astype(np.uint32)
            labels = (
                channels[:, :, 0] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 2]
            )
            lookup = np.full(self._scene.ngeom + 1, -1, dtype=np.int32)
            for geom in self._scene.geoms[: self._scene.ngeom]:
                if geom.segid >= 0 and geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    lookup[geom.segid + 1] = geom.objid
            ids = np.flipud(lookup[labels]).copy()
            ids.flags.writeable = False
        finally:
            self._scene.flags[:] = flags
        return ids, {key: snapshot[key] for key in ("state_hash", "state_time")}

    def _batch(
        self, names: tuple[str, ...]
    ) -> tuple[dict[str, CameraFrame], dict[str, Any]]:
        snapshot = self._snapshot()
        generation = self._generation + 1
        frames = {}
        for name in names:
            started = time.perf_counter()
            rgb = self._draw(name)
            expected = (self.specs[name].height, self.specs[name].width, 3)
            if (
                not isinstance(rgb, np.ndarray)
                or rgb.shape != expected
                or rgb.dtype != np.uint8
                or not rgb.flags.c_contiguous
            ):
                raise RuntimeError(
                    f"invalid RGB frame from {name}; observation not committed"
                )
            finished = time.perf_counter()
            position, rotation = self._pose(name)
            frames[name] = CameraFrame(
                rgb,
                {
                    "frame_generation": generation,
                    "sim_time": snapshot["state_time"],
                    "state_time": snapshot["state_time"],
                    "state_hash": snapshot["state_hash"],
                    "render_started_host_monotonic_s": started,
                    "render_finished_host_monotonic_s": finished,
                    "render_elapsed_s": finished - started,
                    "timestamp_source": "simulation_state_time",
                    "physical_exposure": False,
                    "width": self.specs[name].width,
                    "height": self.specs[name].height,
                    "camera_world_position_m": position.tolist(),
                    "camera_to_world_rotation": rotation.tolist(),
                },
            )
        self._generation = (
            generation  # No partial observation/commit if a render fails.
        )
        return frames, snapshot

    def render(self, name: str) -> CameraFrame:
        if name not in self.specs:
            raise ValueError(f"unknown camera {name!r}")
        frames, _ = self._batch((name,))
        return frames[name]

    def observe(self, *, instruction: str | None = None) -> SimObservation:
        frames, snapshot = self._batch(CAMERA_NAMES)
        for values in snapshot["state"].values():
            values.flags.writeable = False
        return SimObservation(
            images={name: frame.rgb for name, frame in frames.items()},
            state=snapshot["state"],
            image_times={name: snapshot["state_time"] for name in CAMERA_NAMES},
            state_time=snapshot["state_time"],
            instruction=instruction,
            extra={
                "nova_mujoco": {
                    "simulated": True,
                    "controlled_frame": self.controlled_frame,
                    "frame_generation": self._generation,
                    "state_hash": snapshot["state_hash"],
                    "timestamp_source": "simulation_state_time",
                    "camera": {
                        "frames": {
                            name: frame.metadata for name, frame in frames.items()
                        },
                        "same_state_snapshot": True,
                        "physical_exposure": False,
                    },
                }
            },
        )

    def frustum_lines(
        self, name: str, distance_m: float = 0.25
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """World-space frustum edges for offline placement diagnostics."""
        if not math.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("frustum distance must be finite and positive")
        self._snapshot()
        position, rotation = self._pose(name)
        spec = self.specs[name]
        half_h = distance_m * math.tan(math.radians(spec.fovy_deg) / 2)
        half_w = half_h * spec.width / spec.height
        corners = [
            position + rotation @ np.array([x, y, -distance_m])
            for x, y in (
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            )
        ]
        return [(position, corner) for corner in corners] + [
            (corners[i], corners[(i + 1) % 4]) for i in range(4)
        ]

    def render_debug(self) -> np.ndarray:
        """Optional headless overview with frustums and RGB camera axes."""
        self._snapshot()
        self._gl.make_current()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.30, -0.30, 0.2]
        camera.distance = 2.8
        camera.azimuth = 135
        camera.elevation = -30
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self._options,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        for name in CAMERA_NAMES:
            position, rotation = self._pose(name)
            lines = [(a, b, (1.0, 0.8, 0.2, 1.0)) for a, b in self.frustum_lines(name)]
            for i, color in enumerate(
                ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.3, 1.0, 1.0))
            ):
                lines.append((position, position + 0.08 * rotation[:, i], color))
            for start, end, color in lines:
                if self._scene.ngeom >= self._scene.maxgeom:
                    raise RuntimeError("debug scene exceeded max_geoms")
                geom = self._scene.geoms[self._scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array(color, dtype=np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
                self._scene.ngeom += 1
        width, height = (
            self.model.vis.global_.offwidth,
            self.model.vis.global_.offheight,
        )
        rect = mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(rect, self._scene, self._context)
        mujoco.mjr_readPixels(rgb, None, rect, self._context)
        return np.flipud(rgb).copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._gl is not None:
            try:
                self._gl.make_current()
                if self._context is not None:
                    self._context.free()
            finally:
                self._gl.free()
                self._gl = None
                self._context = None

    def __enter__(self) -> NovaMujocoCameraRig:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
