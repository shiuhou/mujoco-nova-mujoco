"""Generic Inspect Robots embodiment over the unchanged S3/S4 simulation."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

from cartesian_bridge import CartesianIKSolver
from manipulation_simulator import ManipulationChunkExecutor, ManipulationSimulator
from nova_cameras import CAMERA_NAMES, NovaMujocoCameraRig
from nova_simulator import Pose
from s5_common import ROOT, Budget, S5Error

SHAPES = dict(zip(CAMERA_NAMES, ((480, 640, 3), (480, 640, 3), (720, 1280, 3))))
LABELS = ("x", "y", "z", "yaw", "pitch", "roll", "gripper")


@dataclass(frozen=True)
class ActionLimits:
    """Simulation engineering bounds, independent of hidden object positions."""

    workspace_low: tuple[float, ...] = (-0.65, -0.45, 0.015)
    workspace_high: tuple[float, ...] = (0.05, 0.35, 0.45)
    waypoint_displacement_m: float = 0.02
    chunk_path_m: float = 0.01
    episode_path_m: float = 0.02
    max_waypoints: int = 32
    open_only: bool = True  # S5A cannot grasp.

    def __post_init__(self):
        low, high = np.asarray(self.workspace_low), np.asarray(self.workspace_high)
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.all(np.isfinite([low, high]))
            or not np.all(low < high)
        ):
            raise ValueError("workspace must have three ordered finite bounds")
        for name in ("waypoint_displacement_m", "chunk_path_m", "episode_path_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("path limits must be finite and positive")
        if self.waypoint_displacement_m > 0.02:
            raise ValueError("cannot relax the existing 0.02 m waypoint boundary")
        if type(self.max_waypoints) is not int or not 1 <= self.max_waypoints <= 64:
            raise ValueError("max_waypoints must be an integer in [1,64]")


class TimedSimulator(ManipulationSimulator):
    """Measure existing deterministic settle/hold work without changing physics."""

    settle_seconds = 0.0

    def settle(self, **kwargs):
        started = time.perf_counter()
        try:
            return super().settle(**kwargs)
        finally:
            self.settle_seconds += time.perf_counter() - started

    def hold(self, duration_s):
        started = time.perf_counter()
        try:
            return super().hold(duration_s)
        finally:
            self.settle_seconds += time.perf_counter() - started


class TimedIK(CartesianIKSolver):
    seconds = 0.0

    def solve(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return super().solve(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - started


class NovaMujocoEmbodiment(EmbodimentBase):
    """Single-owner observe -> policy -> validated chunk -> settle -> observe."""

    def __init__(self, budget: Budget, limits: ActionLimits | None = None):
        self.budget = budget
        self.limits = limits or ActionLimits()
        self.sim = TimedSimulator()
        self.executor = ManipulationChunkExecutor(self.sim)
        self.executor.ik = TimedIK(self.executor.kinematics)
        try:
            self.rig = NovaMujocoCameraRig(
                self.sim, config_path=ROOT / "configs/s3_s4_cameras.json"
            )
        except BaseException:
            self.executor.close()
            self.sim.close()
            raise
        self.info = EmbodimentInfo(
            name="nova2_mujoco_simulation_only",
            action_space=Box(
                shape=(7,),
                low=np.array([*self.limits.workspace_low, 0, 0, 0, 0.0]),
                high=np.array([*self.limits.workspace_high, 0, 0, 0, 1.0]),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    # The pinned official agent rejects Euler rotational
                    # interpolation. These three public slots are fixed at 0;
                    # advertise no active rotation, not rot6d or arbitrary RPY.
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=LABELS,
                    # sqrt(3)*.005 < the executor's .020 metre bound.
                    max_step=(0.005, 0.005, 0.005, None, None, None, 0.2),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(n, *SHAPES[n][:2]) for n in CAMERA_NAMES),
                state=StateSpec((StateField("eef_state", (7,), "m,rad,normalized"),)),
            ),
            control_hz=10.0,  # Interpolation metadata; no high-rate model inference.
            is_simulated=True,
            capabilities=frozenset({"resettable", "renderable"}),
            environment_id="nova2_mujoco",
            environment_revision="benchmark_v0+s3_s4_cameras",
            docs=(
                "This embodiment is SIMULATION ONLY; no real robot is connected. "
                "move_to targets are absolute base-frame x,y,z metres at the "
                "nominal grasp center, flange +Z 197 mm (not a measured TCP). "
                "The public order is x,y,z,yaw,pitch,roll,gripper. Rotations are "
                "reset-relative radians; only yaw=pitch=roll=0 is supported. "
                "Nonzero orientation is rejected. Gripper 0=closed, 1=open; "
                "maximum opening is 95 mm. eef_state reports measured aperture, "
                "including small contact/compliance deviations at endpoints. "
                + (
                    "S5A requires gripper=1. "
                    if self.limits.open_only
                    else "Gripper commands in [0,1] are supported. "
                )
                + f"Each decision permits at most {self.limits.chunk_path_m:.3f} m Cartesian path and {self.limits.max_waypoints} interpolated waypoints. "
                "All actions are validated without clamping by this "
                "backend. A failure aborts the remaining chunk. The controller "
                "executes bounded waypoints, settles, and supplies fresh images "
                "before the next decision. This is not calibrated robot dynamics."
            ),
        )
        self.current: Observation | None = None
        self._current_meta: dict = {}
        self._generation = 0
        self._inflight = False
        self._scene: Scene | None = None
        self.path_m = 0.0
        self.observation_trace: list[dict] = []
        self.action_trace: list[dict] = []
        self.validation_trace: list[dict] = []
        self.timing = dict.fromkeys(
            ("T_render", "T_chunk_validation", "T_IK", "T_execute", "T_settle"), 0.0
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        if seed is not None:
            raise ValueError("benchmark_v0 has a fixed initial scene, not random seeds")
        self.current = None
        self._inflight = False
        self.sim.reset()
        self.executor.reset_reference()
        self._scene = scene
        self.path_m = 0.0
        return self.observe()

    def observe(self) -> Observation:
        self.budget.remaining()
        self.current = None  # Failed rendering must never leave a reusable frame.
        self._inflight = False
        started = time.perf_counter()
        try:
            raw = self.rig.observe(
                instruction=self._scene.instruction if self._scene else None
            )
        except S5Error:
            raise
        except Exception:  # noqa: BLE001 - never persist arbitrary renderer exceptions
            raise S5Error("observation_failure", "required RGB render failed") from None
        finally:
            self.timing["T_render"] += time.perf_counter() - started
        meta = raw.extra["nova_mujoco"]
        valid = (
            tuple(raw.images) == CAMERA_NAMES
            and meta["frame_generation"] > self._generation
            and raw.state["eef_state"].shape == (7,)
            and np.all(np.isfinite(raw.state["eef_state"]))
        )
        for name, shape in SHAPES.items():
            image = raw.images[name]
            frame = meta["camera"]["frames"][name]
            valid = valid and (
                image.shape == shape
                and image.dtype == np.uint8
                and image.flags.c_contiguous
                and not image.flags.writeable
                and frame["state_hash"] == meta["state_hash"]
                and frame["frame_generation"] == meta["frame_generation"]
                and frame["state_time"] == raw.state_time == raw.image_times[name]
            )
        if not valid:
            raise S5Error("observation_failure", "incoherent camera/state bundle")
        self._generation = meta["frame_generation"]
        self._current_meta = meta
        # Explicit allowlist. No joints, simulator metadata, geometry, target,
        # contacts, segmentation, object coordinates or success reach the policy.
        obs = Observation(
            images=raw.images,
            state={"eef_state": raw.state["eef_state"]},
            instruction=raw.instruction,
            image_times=raw.image_times,
            state_time=raw.state_time,
            extra={"env_step": self.budget.steps},
        )
        self.observation_trace.append(
            {
                "generation": self._generation,
                "state_hash": meta["state_hash"],
                "simulation_time_s": raw.state_time,
                "eef_state": obs.state["eef_state"].tolist(),
                "images": {
                    name: {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "order": "RGB",
                        "sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "generation": meta["frame_generation"],
                        "state_hash": meta["state_hash"],
                    }
                    for name, rgb in obs.images.items()
                },
            }
        )
        self.current = obs
        return obs

    def _assert_current(self, obs: Observation) -> None:
        if obs is not self.current or self.current is None:
            raise S5Error(
                "stale_observation", "observation is not the current owned bundle"
            )
        # Reuse the S4 snapshot/hash implementation. It copies render data only.
        if self.rig._snapshot()["state_hash"] != self._current_meta["state_hash"]:
            self.current = None
            raise S5Error("stale_observation", "simulation changed after observation")

    def begin_decision(self, obs: Observation) -> None:
        self.budget.remaining()
        self._assert_current(obs)
        if self._inflight:
            raise S5Error("stale_observation", "bundle already consumed by a decision")
        self._inflight = True

    def validate_waypoints(self, waypoints) -> np.ndarray:
        started = time.perf_counter()
        ik_before = self.executor.ik.seconds
        category = "accepted"
        try:
            if not self._inflight or self.current is None:
                raise S5Error(
                    "stale_observation", "no active observation/decision boundary"
                )
            self._assert_current(self.current)
            try:
                values = np.asarray(waypoints, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise S5Error(
                    "invalid_action", "chunk must contain numeric 7D actions"
                ) from None
            if (
                values.ndim != 2
                or values.shape[1] != 7
                or not 1 <= len(values) <= self.limits.max_waypoints
                or not np.all(np.isfinite(values))
            ):
                raise S5Error(
                    "invalid_action", "invalid finite (N,7) chunk shape/length"
                )
            self.budget.check_steps(len(values))
            if np.any(values[:, 3:6] != 0):
                raise S5Error("invalid_action", "unsupported nonzero orientation")
            if np.any((values[:, 6] < 0) | (values[:, 6] > 1)):
                raise S5Error("invalid_action", "gripper outside [0,1]")
            # Official interpolation starts at measured aperture. Preserve tiny
            # numerical compliance rather than rewriting observations/actions;
            # require the final open command exactly and intermediate error <1e-8.
            if self.limits.open_only and (
                np.any(np.abs(values[:, 6] - 1) > 1e-8) or values[-1, 6] != 1
            ):
                raise S5Error(
                    "safety_validation_abort", "S5A requires gripper fully open"
                )
            if np.any(values[:, :3] < self.limits.workspace_low) or np.any(
                values[:, :3] > self.limits.workspace_high
            ):
                raise S5Error(
                    "workspace_reject", "target outside declared simulation workspace"
                )
            start = self.sim.get_tool_pose().position_m
            distances = np.linalg.norm(
                np.diff(np.vstack([start, values[:, :3]]), axis=0), axis=1
            )
            path = float(distances.sum())
            if np.any(distances > self.limits.waypoint_displacement_m + 1e-12):
                raise S5Error("invalid_action", "waypoint displacement exceeds limit")
            if (
                path > self.limits.chunk_path_m + 1e-12
                or self.path_m + path > self.limits.episode_path_m + 1e-12
            ):
                raise S5Error(
                    "safety_validation_abort", "Cartesian path budget exceeded"
                )
            # Preflight every waypoint in order on isolated kinematic data.
            # Live object state/contact history cannot change during this check.
            seed = self.sim.get_joint_position()
            for vector in values:
                self.executor.kinematics.reset(seed)
                result = self.executor.ik.solve(
                    Pose(vector[:3], self.executor.reference_rotation),
                    frame=self.executor.frame,
                )
                if not result.converged:
                    raise S5Error(
                        "IK_fail", "waypoint inverse kinematics did not converge"
                    )
                seed = result.q_rad
            return values.copy()
        except S5Error as exc:
            category = exc.category
            raise
        finally:
            ik = self.executor.ik.seconds - ik_before
            self.timing["T_IK"] += ik
            self.timing["T_chunk_validation"] += max(
                0, time.perf_counter() - started - ik
            )
            self.validation_trace.append(
                {"generation": self._generation, "result": category}
            )

    def pre_check(self, waypoints) -> str | None:
        try:
            self.validate_waypoints(waypoints)
        except S5Error as exc:
            return str(exc)  # Only our fixed, nonprivileged validation reasons.
        return None

    def execute_chunk(self, chunk: ActionChunk, obs: Observation) -> StepResult:
        self._assert_current(obs)
        if not isinstance(chunk, ActionChunk):
            raise S5Error("invalid_action", "official ActionChunk required")
        if any(a.meta.get("request_stop") for a in chunk.actions):
            raise S5Error("invalid_action", "terminal tool output is not motion")
        values = self.validate_waypoints([a.data for a in chunk.actions])
        start = self.sim.get_tool_pose().position_m.copy()
        generation = self._generation
        self.current = None
        self._inflight = False
        # Freeze the validated order/data against subsequent mutation by a caller.
        frozen = ActionChunk(
            actions=tuple(Action(data=v.copy()) for v in values),
            control_hz=chunk.control_hz,
        )
        started = time.perf_counter()
        ik_before = self.executor.ik.seconds
        settle_before = self.sim.settle_seconds
        result = None
        try:
            result = self.executor.execute_chunk(frozen)
            self.budget.steps += len(result.records)
            previous = start
            for record in result.records:
                self.path_m += float(
                    np.linalg.norm(record.achieved_pose.position_m - previous)
                )
                previous = record.achieved_pose.position_m
            if result.status != "completed":
                category = (
                    "IK_fail"
                    if "IK_fail" in (result.reason or "")
                    else "safety_validation_abort"
                )
                raise S5Error(category, "executor rejected waypoint; remainder stopped")
            # Gripper servo can still be slewing after the arm's settle. Wait a
            # deterministic aperture-derived duration plus a fixed final hold.
            gs = self.sim.get_gripper_state()
            duration = (
                abs(gs["commanded_opening_m"] - gs["measured_opening_m"])
                / self.sim.scene_spec.gripper.opening_speed_m_s
            )
            self.sim.hold(duration + 0.25)
            self.budget.remaining()
        finally:
            ik = self.executor.ik.seconds - ik_before
            settle = self.sim.settle_seconds - settle_before
            self.timing["T_IK"] += ik
            self.timing["T_settle"] += settle
            self.timing["T_execute"] += max(
                0, time.perf_counter() - started - ik - settle
            )
            self.action_trace.append(
                {
                    "observation_generation": generation,
                    "ordered_actions": values.tolist(),
                    "executed_waypoints": 0 if result is None else len(result.records),
                    "waypoint_results": []
                    if result is None
                    else [
                        {
                            "index": record.index,
                            "requested_action": record.requested_action.tolist(),
                            "achieved_position_m": record.achieved_pose.position_m.tolist(),
                            "IK_status": record.ik.status,
                            "IK_iterations": record.ik.iterations,
                            "position_error_m": record.ik.position_error_m,
                            "orientation_error_rad": record.ik.orientation_error_rad,
                        }
                        for record in result.records
                    ],
                    "failed_index": None if result is None else result.failed_index,
                    "executor_status": "exception" if result is None else result.status,
                    "executor": "ManipulationChunkExecutor(NovaMujocoChunkExecutor)",
                    "simulation_time_after_settle_s": self.sim.simulation_time_s,
                }
            )
        fresh = self.observe()
        return StepResult(observation=fresh)

    def step(self, action: Action) -> StepResult:
        if self.current is None:
            raise S5Error("stale_observation", "reset/observe required before step")
        obs = self.current
        self.begin_decision(obs)
        return self.execute_chunk(ActionChunk(actions=(action,)), obs)

    def close(self) -> None:
        self.current = None
        self.rig.close()
        self.executor.close()
        self.sim.close()
