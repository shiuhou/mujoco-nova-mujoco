"""Deterministic, hardware-independent Nova2 MuJoCo execution primitives.

The required S1 baseline is a kinematic reference executor.  It advances an
explicit simulation clock and never sleeps or consults wall time.  It makes no
claim about Nova controller, motor, payload, friction, or contact dynamics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from nova_fk import JOINT_NAMES, MODEL_PATH

FloatArray = npt.NDArray[np.float64]
FRAME_LINK6 = "link6"
FRAME_NOMINAL_GRASP_CENTER = "nominal_grasp_center"
SUPPORTED_FRAMES = (FRAME_LINK6, FRAME_NOMINAL_GRASP_CENTER)


class SimulatorError(RuntimeError):
    """Base error for deterministic simulator execution."""


class JointTargetError(ValueError):
    """A joint vector is malformed or outside the model limits."""


@dataclass(frozen=True)
class SimulatorConfig:
    """SIMULATION_ENGINEERING_PARAMETERS, not measured Nova parameters."""

    timestep_s: float = 0.002
    max_joint_speed_rad_s: float = 1.0
    default_frame: str = FRAME_LINK6

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestep_s) or self.timestep_s <= 0:
            raise ValueError("timestep_s must be finite and positive")
        if (
            not math.isfinite(self.max_joint_speed_rad_s)
            or self.max_joint_speed_rad_s <= 0
        ):
            raise ValueError("max_joint_speed_rad_s must be finite and positive")
        if self.default_frame not in SUPPORTED_FRAMES:
            raise ValueError(f"default_frame must be one of {SUPPORTED_FRAMES}")


@dataclass(frozen=True)
class Pose:
    position_m: FloatArray
    rotation_matrix: FloatArray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position_m": self.position_m.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
        }


@dataclass(frozen=True)
class ContactDiagnostic:
    geom1: str
    geom2: str
    distance_m: float


@dataclass(frozen=True)
class JointCommand:
    start_q_rad: FloatArray
    target_q_rad: FloatArray
    total_steps: int


def _finite_vector(
    value: Sequence[float] | npt.ArrayLike, length: int, label: str
) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain exactly {length} finite values")
    return result


class NovaMujocoSimulator:
    """Small deterministic Nova2 simulator with a kinematic reference mode."""

    mode = "kinematic"

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        *,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.config = config or SimulatorConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = self.config.timestep_s
        self.data = mujoco.MjData(self.model)
        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=int,
        )
        if np.any(self._joint_ids < 0):
            raise SimulatorError("model is missing one or more Nova2 joints")
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids].copy()
        self._dof_addresses = self.model.jnt_dofadr[self._joint_ids].copy()
        if not np.all(self.model.jnt_limited[self._joint_ids]):
            raise SimulatorError("all Nova2 joints must have model-defined limits")
        self._joint_limits = self.model.jnt_range[self._joint_ids].copy()
        self._base_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._site_ids = {
            FRAME_LINK6: self._require_id(mujoco.mjtObj.mjOBJ_SITE, "tcp"),
            FRAME_NOMINAL_GRASP_CENTER: self._require_id(
                mujoco.mjtObj.mjOBJ_SITE, "nominal_grasp_center"
            ),
        }
        self._pending: JointCommand | None = None
        self._pending_step = 0
        self._closed = False
        self.reset(np.zeros(6, dtype=np.float64))

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise SimulatorError(f"model is missing {name!r}")
        return object_id

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def simulation_time_s(self) -> float:
        return float(self.data.time)

    @property
    def joint_limits(self) -> FloatArray:
        return self._joint_limits.copy()

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    @property
    def has_pending_command(self) -> bool:
        return self._pending is not None

    @property
    def reference_rotation(self) -> FloatArray:
        return self._reference_rotation.copy()

    def _ensure_open(self) -> None:
        if self._closed:
            raise SimulatorError("simulator is closed")

    def validate_joint_target(self, q: Sequence[float] | npt.ArrayLike) -> FloatArray:
        try:
            target = _finite_vector(q, 6, "joint target")
        except ValueError as exc:
            raise JointTargetError(str(exc)) from exc
        below = target < self._joint_limits[:, 0]
        above = target > self._joint_limits[:, 1]
        if np.any(below | above):
            index = int(np.flatnonzero(below | above)[0])
            low, high = self._joint_limits[index]
            raise JointTargetError(
                f"joint{index + 1} target {target[index]:.12g} rad is outside "
                f"model range [{low:.12g}, {high:.12g}]"
            )
        return target.copy()

    def reset(self, q: Sequence[float] | npt.ArrayLike | None = None) -> None:
        self._ensure_open()
        target = (
            np.zeros(6, dtype=np.float64)
            if q is None
            else self.validate_joint_target(q)
        )
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = target
        self.data.qvel[self._dof_addresses] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._pending = None
        self._pending_step = 0
        self._reference_rotation = self.get_tool_pose().rotation_matrix

    def get_joint_position(self) -> FloatArray:
        self._ensure_open()
        return self.data.qpos[self._qpos_addresses].copy()

    def get_link6_pose(self) -> Pose:
        return self.get_tool_pose(FRAME_LINK6)

    def get_tool_pose(self, frame: str | None = None) -> Pose:
        self._ensure_open()
        selected = frame or self.config.default_frame
        if selected not in self._site_ids:
            raise ValueError(f"frame must be one of {SUPPORTED_FRAMES}")
        site_id = self._site_ids[selected]
        base_rotation = self.data.xmat[self._base_id].reshape(3, 3)
        position = base_rotation.T @ (
            self.data.site_xpos[site_id] - self.data.xpos[self._base_id]
        )
        rotation = base_rotation.T @ self.data.site_xmat[site_id].reshape(3, 3)
        return Pose(position.copy(), rotation.copy())

    def site_jacobian(self, frame: str | None = None) -> tuple[FloatArray, FloatArray]:
        self._ensure_open()
        selected = frame or self.config.default_frame
        if selected not in self._site_ids:
            raise ValueError(f"frame must be one of {SUPPORTED_FRAMES}")
        jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jac_position,
            jac_rotation,
            self._site_ids[selected],
        )
        columns = self._dof_addresses
        # mj_jacSite returns world-coordinate derivatives. get_tool_pose and
        # the Cartesian solver's residual both use base coordinates.
        world_to_base = self.data.xmat[self._base_id].reshape(3, 3).T
        return world_to_base @ jac_position[:, columns], world_to_base @ jac_rotation[
            :, columns
        ]

    def set_joint_state_for_kinematics(
        self, q: Sequence[float] | npt.ArrayLike
    ) -> None:
        """Set a validated state without advancing time; used by bounded IK iterations."""
        self._ensure_open()
        target = self.validate_joint_target(q)
        self.data.qpos[self._qpos_addresses] = target
        self.data.qvel[self._dof_addresses] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def command_joint_target(
        self,
        q_target: Sequence[float] | npt.ArrayLike,
        *,
        duration_s: float | None = None,
    ) -> JointCommand:
        self._ensure_open()
        if self._pending is not None:
            raise SimulatorError("a joint command is already pending")
        target = self.validate_joint_target(q_target)
        start = self.get_joint_position()
        maximum_delta = float(np.max(np.abs(target - start)))
        required_duration = maximum_delta / self.config.max_joint_speed_rad_s
        if duration_s is None:
            duration = required_duration
        else:
            if not math.isfinite(duration_s) or duration_s < 0:
                raise ValueError("duration_s must be finite and non-negative")
            if duration_s + 1e-15 < required_duration:
                raise JointTargetError(
                    f"duration_s={duration_s:.12g} exceeds configured joint speed limit"
                )
            duration = duration_s
        ratio = duration / self.timestep_s
        steps = max(1, math.ceil(ratio - 1e-12 * max(1.0, abs(ratio))))
        command = JointCommand(start, target, steps)
        self._pending = command
        self._pending_step = 0
        return command

    def step(self, steps: int = 1) -> int:
        """Advance a fixed number of simulation ticks without wall-clock waiting."""
        self._ensure_open()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        advanced = 0
        for _ in range(steps):
            if self._pending is not None:
                self._pending_step += 1
                fraction = min(1.0, self._pending_step / self._pending.total_steps)
                q = self._pending.start_q_rad + fraction * (
                    self._pending.target_q_rad - self._pending.start_q_rad
                )
                self.data.qpos[self._qpos_addresses] = q
                self.data.qvel[self._dof_addresses] = 0.0
                if self._pending_step >= self._pending.total_steps:
                    self._pending = None
                    self._pending_step = 0
            self.data.time += self.timestep_s
            mujoco.mj_forward(self.model, self.data)
            advanced += 1
        return advanced

    def settle(self, *, max_steps: int = 100_000) -> int:
        self._ensure_open()
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        advanced = 0
        while self._pending is not None and advanced < max_steps:
            self.step()
            advanced += 1
        if self._pending is not None:
            raise SimulatorError(
                f"joint command did not settle within {max_steps} steps"
            )
        return advanced

    def contact_diagnostics(self) -> tuple[ContactDiagnostic, ...]:
        self._ensure_open()
        result: list[ContactDiagnostic] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            result.append(
                ContactDiagnostic(
                    geom1 or "<unnamed>", geom2 or "<unnamed>", contact.dist
                )
            )
        return tuple(result)

    def get_robot_state(self, frame: str | None = None) -> dict[str, object]:
        pose = self.get_tool_pose(frame)
        return {
            "joint_pos": self.get_joint_position(),
            "eef_position_m": pose.position_m,
            "eef_rotation_matrix": pose.rotation_matrix,
            "simulation_time_s": self.simulation_time_s,
            "logical_camera_names": ("front_rgb", "right_rgb", "wrist_rgb"),
        }

    def close(self) -> None:
        self._pending = None
        self._closed = True

    def __enter__(self) -> NovaMujocoSimulator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
