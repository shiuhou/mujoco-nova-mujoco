"""Bounded Cartesian IK and Robocurve-compatible waypoint execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from nova_simulator import NovaMujocoSimulator, Pose

FloatArray = npt.NDArray[np.float64]


class CartesianTargetError(ValueError):
    """The generic Cartesian target violates the S2 action contract."""


class IKFailure(RuntimeError):
    """The bounded IK solver did not converge to the requested pose."""


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 200
    position_tolerance_m: float = 1e-6
    orientation_tolerance_rad: float = 1e-6
    damping: float = 1e-4
    max_joint_update_rad: float = 0.08
    stagnation_iterations: int = 20

    def __post_init__(self) -> None:
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        for name in (
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "damping",
            "max_joint_update_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.stagnation_iterations) is not int
            or self.stagnation_iterations <= 0
        ):
            raise ValueError("stagnation_iterations must be positive")


@dataclass(frozen=True)
class IKResult:
    status: str
    q_rad: FloatArray
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    joint_displacement_rad: float
    condition_number: float

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "q_rad": self.q_rad.tolist(),
            "iterations": self.iterations,
            "position_error_m": self.position_error_m,
            "position_error_mm": self.position_error_m * 1000.0,
            "orientation_error_rad": self.orientation_error_rad,
            "orientation_error_deg": math.degrees(self.orientation_error_rad),
            "joint_displacement_rad": self.joint_displacement_rad,
            "condition_number": self.condition_number,
        }


@dataclass(frozen=True)
class WaypointRecord:
    index: int
    requested_action: FloatArray
    target_pose: Pose
    ik: IKResult
    achieved_q_rad: FloatArray
    achieved_pose: Pose
    contacts: tuple[dict[str, object], ...]
    simulation_time_s: float

    def as_dict(self) -> dict[str, object]:
        delta = self.achieved_pose.position_m - self.target_pose.position_m
        return {
            "index": self.index,
            "requested_action": self.requested_action.tolist(),
            "target_pose": self.target_pose.as_dict(),
            "ik": self.ik.as_dict(),
            "achieved_q_rad": self.achieved_q_rad.tolist(),
            "achieved_pose": self.achieved_pose.as_dict(),
            "position_residual_m": delta.tolist(),
            "contacts": list(self.contacts),
            "simulation_time_s": self.simulation_time_s,
        }


@dataclass(frozen=True)
class ChunkExecutionResult:
    status: str
    records: tuple[WaypointRecord, ...]
    failed_index: int | None = None
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


def rotation_error_vector(target: npt.ArrayLike, current: npt.ArrayLike) -> FloatArray:
    """World-frame log map for ``target @ current.T``."""
    relative = (
        np.asarray(target, dtype=np.float64) @ np.asarray(current, dtype=np.float64).T
    )
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    sine = 0.5 * float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.atan2(sine, cosine)
    if angle < 1e-14:
        return np.zeros(3, dtype=np.float64)
    if sine < 1e-10:
        values, vectors = np.linalg.eigh(relative + relative.T)
        axis = vectors[:, int(np.argmax(values))]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        return axis * angle
    return skew * (angle / (2.0 * sine))


def rotation_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    return float(np.linalg.norm(rotation_error_vector(a, b)))


class CartesianIKSolver:
    """Deterministic damped least-squares IK using MuJoCo site Jacobians."""

    def __init__(
        self, simulator: NovaMujocoSimulator, *, config: IKConfig | None = None
    ) -> None:
        self.simulator = simulator
        self.config = config or IKConfig()

    def solve(
        self,
        target_pose: Pose,
        *,
        frame: str | None = None,
        seed_q: Sequence[float] | npt.ArrayLike | None = None,
    ) -> IKResult:
        position = np.asarray(target_pose.position_m, dtype=np.float64)
        rotation = np.asarray(target_pose.rotation_matrix, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise CartesianTargetError(
                "target position must contain three finite metres"
            )
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
        ):
            raise CartesianTargetError(
                "target orientation must be a proper finite rotation"
            )
        original_q = self.simulator.get_joint_position()
        q = (
            original_q
            if seed_q is None
            else self.simulator.validate_joint_target(seed_q)
        )
        limits = self.simulator.joint_limits
        best_error = math.inf
        stagnant = 0
        condition = math.inf
        try:
            for iteration in range(self.config.max_iterations + 1):
                self.simulator.set_joint_state_for_kinematics(q)
                current = self.simulator.get_tool_pose(frame)
                position_error = position - current.position_m
                orientation_error = rotation_error_vector(
                    rotation, current.rotation_matrix
                )
                position_norm = float(np.linalg.norm(position_error))
                orientation_norm = float(np.linalg.norm(orientation_error))
                if (
                    position_norm <= self.config.position_tolerance_m
                    and orientation_norm <= self.config.orientation_tolerance_rad
                ):
                    return IKResult(
                        "converged",
                        q.copy(),
                        iteration,
                        position_norm,
                        orientation_norm,
                        float(np.linalg.norm(q - original_q)),
                        condition,
                    )
                if iteration == self.config.max_iterations:
                    break
                jac_position, jac_rotation = self.simulator.site_jacobian(frame)
                jacobian = np.vstack((jac_position, jac_rotation))
                singular_values = np.linalg.svd(jacobian, compute_uv=False)
                condition = (
                    math.inf
                    if singular_values[-1] <= 1e-14
                    else float(singular_values[0] / singular_values[-1])
                )
                error = np.concatenate((position_error, orientation_error))
                system = jacobian @ jacobian.T + self.config.damping**2 * np.eye(6)
                delta = jacobian.T @ np.linalg.solve(system, error)
                maximum = float(np.max(np.abs(delta)))
                if maximum > self.config.max_joint_update_rad:
                    delta *= self.config.max_joint_update_rad / maximum
                candidate = q + delta
                if np.any(candidate < limits[:, 0]) or np.any(candidate > limits[:, 1]):
                    return IKResult(
                        "joint_limit",
                        q.copy(),
                        iteration + 1,
                        position_norm,
                        orientation_norm,
                        float(np.linalg.norm(q - original_q)),
                        condition,
                    )
                total_error = position_norm + orientation_norm
                if best_error - total_error <= 1e-12:
                    stagnant += 1
                else:
                    stagnant = 0
                    best_error = total_error
                if stagnant >= self.config.stagnation_iterations:
                    return IKResult(
                        "stagnated",
                        q.copy(),
                        iteration + 1,
                        position_norm,
                        orientation_norm,
                        float(np.linalg.norm(q - original_q)),
                        condition,
                    )
                q = candidate
            current = self.simulator.get_tool_pose(frame)
            return IKResult(
                "max_iterations",
                q.copy(),
                self.config.max_iterations,
                float(np.linalg.norm(position - current.position_m)),
                rotation_distance(rotation, current.rotation_matrix),
                float(np.linalg.norm(q - original_q)),
                condition,
            )
        finally:
            self.simulator.set_joint_state_for_kinematics(original_q)


class NovaMujocoChunkExecutor:
    """Thin public-data protocol boundary for ActionChunk-style waypoints.

    Each waypoint is seven finite values in the verified public order
    ``[x,y,z,yaw,pitch,roll,gripper]``.  This S2 implementation supports
    translation only: relative orientation and inactive gripper must all be 0.
    """

    def __init__(
        self,
        simulator: NovaMujocoSimulator,
        *,
        frame: str | None = None,
        ik_config: IKConfig | None = None,
        max_cartesian_step_m: float = 0.02,
    ) -> None:
        if not math.isfinite(max_cartesian_step_m) or max_cartesian_step_m <= 0:
            raise ValueError("max_cartesian_step_m must be finite and positive")
        self.simulator = simulator
        self.frame = frame or simulator.config.default_frame
        self.ik = CartesianIKSolver(simulator, config=ik_config)
        self.max_cartesian_step_m = max_cartesian_step_m
        self.reference_rotation = simulator.reference_rotation

    def reset_reference(self) -> None:
        self.reference_rotation = self.simulator.get_tool_pose(
            self.frame
        ).rotation_matrix

    @staticmethod
    def _action_data(action: Any) -> FloatArray:
        value = getattr(action, "data", action)
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CartesianTargetError("action must be a numeric SI vector") from exc
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise CartesianTargetError(
                "action must be finite [x,y,z,yaw,pitch,roll,gripper] with shape (7,)"
            )
        if not np.array_equal(vector[3:6], np.zeros(3)):
            raise CartesianTargetError(
                "S2 orientation is pinned; yaw, pitch, and roll must be 0"
            )
        if vector[6] != 0:
            raise CartesianTargetError("gripper is inactive; its only legal value is 0")
        return vector.copy()

    @staticmethod
    def _actions(chunk: Any) -> Iterable[Any]:
        actions = getattr(chunk, "actions", chunk)
        try:
            result = tuple(actions)
        except TypeError as exc:
            raise CartesianTargetError(
                "chunk must be an iterable or expose .actions"
            ) from exc
        if not result:
            raise CartesianTargetError("chunk must contain at least one action")
        return result

    def execute_chunk(self, chunk: Any) -> ChunkExecutionResult:
        records: list[WaypointRecord] = []
        for index, raw_action in enumerate(self._actions(chunk)):
            try:
                action = self._action_data(raw_action)
                current_pose = self.simulator.get_tool_pose(self.frame)
                displacement = action[:3] - current_pose.position_m
                distance = float(np.linalg.norm(displacement))
                if distance > self.max_cartesian_step_m + 1e-12:
                    raise CartesianTargetError(
                        f"Cartesian step {distance:.12g} m exceeds "
                        f"{self.max_cartesian_step_m:.12g} m"
                    )
                target = Pose(action[:3].copy(), self.reference_rotation.copy())
                solution = self.ik.solve(target, frame=self.frame)
                if not solution.converged:
                    raise IKFailure(
                        f"IK {solution.status}: position={solution.position_error_m:.6g} m, "
                        f"orientation={solution.orientation_error_rad:.6g} rad"
                    )
                self.simulator.command_joint_target(solution.q_rad)
                self.simulator.settle()
                achieved = self.simulator.get_tool_pose(self.frame)
                contacts = tuple(
                    {
                        "geom1": item.geom1,
                        "geom2": item.geom2,
                        "distance_m": item.distance_m,
                    }
                    for item in self.simulator.contact_diagnostics()
                )
                records.append(
                    WaypointRecord(
                        index,
                        action,
                        target,
                        solution,
                        self.simulator.get_joint_position(),
                        achieved,
                        contacts,
                        self.simulator.simulation_time_s,
                    )
                )
            except (CartesianTargetError, IKFailure, ValueError) as exc:
                return ChunkExecutionResult("rejected", tuple(records), index, str(exc))
        return ChunkExecutionResult("completed", tuple(records))
