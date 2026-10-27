"""S3: prescribed arm references with dynamic fingers and free objects.

The frozen S1/S2 implementation is reused without edits. Only arm qpos/qvel
are prescribed during execution; MuJoCo alone integrates object/finger state.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from benchmark_scene import SCENE_PATH, BenchmarkSceneSpec, benchmark_v0
from cartesian_bridge import (
    CartesianIKSolver,
    CartesianTargetError,
    ChunkExecutionResult,
    IKFailure,
    NovaMujocoChunkExecutor,
    WaypointRecord,
)
from nova_simulator import (
    FRAME_NOMINAL_GRASP_CENTER,
    NovaMujocoSimulator,
    Pose,
    SimulatorConfig,
    SimulatorError,
)


class ManipulationFault(SimulatorError):
    """A latched contact/numerical fault stops execution until explicit reset."""


class ManipulationSimulator(NovaMujocoSimulator):
    mode = "kinematic_arm_with_dynamic_objects"

    def __init__(
        self, model_path: Path = SCENE_PATH, *, spec: BenchmarkSceneSpec | None = None
    ):
        self.scene_spec = spec or benchmark_v0()
        self._ready = False
        super().__init__(
            model_path,
            config=SimulatorConfig(
                timestep_s=self.scene_spec.physics.timestep_s,
                max_joint_speed_rad_s=self.scene_spec.physics.max_joint_speed_rad_s,
                default_frame=FRAME_NOMINAL_GRASP_CENTER,
            ),
        )
        self._finger_joints = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, f"{s}_finger_slide")
                for s in ("left", "right")
            ]
        )
        self._finger_qpos = self.model.jnt_qposadr[self._finger_joints]
        self._actuators = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_finger_drive")
                for s in ("left", "right")
            ]
        )
        self._object_ids = {
            obj.semantic_name: self._require_id(
                mujoco.mjtObj.mjOBJ_BODY, obj.semantic_name
            )
            for obj in self.scene_spec.objects
        }
        self._geom_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            for i in range(self.model.ngeom)
        ]
        self._grasp_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
        self._ready = True
        self.reset()

    def reset(self, q=None) -> None:
        if not self._ready:
            super().reset(q)
            return
        self._ensure_open()
        target = self.validate_joint_target(
            self.scene_spec.initial_q_rad if q is None else q
        )
        # The only operation that restores object poses is explicit scene reset.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self._qpos_addresses] = target
        self._pending = None
        self._pending_step = 0
        self._commanded_opening = (
            self.scene_spec.initial_gripper * self.scene_spec.gripper.max_opening_m
        )
        self._setpoint_opening = self._commanded_opening
        self.data.ctrl[self._actuators] = self._setpoint_opening / 2
        mujoco.mj_forward(self.model, self.data)
        self._reference_rotation = self.get_tool_pose().rotation_matrix
        self.fault: str | None = None
        self._tick = 0
        self._initial_object_positions = {
            name: self.data.xpos[bid].copy() for name, bid in self._object_ids.items()
        }
        self._candidate_relative: np.ndarray | None = None
        self._candidate_time = 0.0
        self._previous_relative: np.ndarray | None = None
        self._slip = 0.0
        self._max_slip = 0.0
        self._max_lift = 0.0
        self._ever_grasped = False
        self._ever_stable = False
        self._holding = False
        self._state = "OPEN" if self.scene_spec.initial_gripper > 0 else "CLOSED_EMPTY"
        self._bilateral = False
        self._finger_contacts: dict[str, list[str]] = {"left": [], "right": []}
        self._settled_time = 0.0
        self._max_black_displacement = 0.0
        self._pair_counts: Counter[str] = Counter()
        self._pair_max_force: dict[str, float] = {}
        self._max_penetration = 0.0
        self._max_arm_projection_error = 0.0
        self.contact_timeline: list[dict] = []
        self._last_pairs: tuple[str, ...] = ()
        self._last_contacts: list[dict] = self.contact_records()

    @staticmethod
    def validate_gripper(value: float) -> float:
        try:
            if np.ndim(value) != 0:
                raise ValueError("gripper must be a scalar")
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper must be a finite scalar in [0, 1]") from exc
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("gripper must be finite and in [0, 1]; 0 closed, 1 open")
        return result

    def command_gripper(self, normalized: float) -> None:
        self._ensure_open()
        if self.fault:
            raise ManipulationFault(self.fault)
        self._commanded_opening = (
            self.validate_gripper(normalized) * self.scene_spec.gripper.max_opening_m
        )

    def command_joint_target(self, q_target, *, duration_s=None):
        if self._ready and self.fault:
            raise ManipulationFault(self.fault)
        return super().command_joint_target(q_target, duration_s=duration_s)

    def _abort(self, reason: str) -> None:
        self.fault = reason
        self._pending = None
        self.data.qvel[self._dof_addresses] = 0
        raise ManipulationFault(reason)

    def step(self, steps: int = 1) -> int:
        self._ensure_open()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self.fault:
            raise ManipulationFault(self.fault)
        for _ in range(steps):
            start = self.get_joint_position()
            target = start.copy()
            if self._pending is not None:
                self._pending_step += 1
                fraction = min(1.0, self._pending_step / self._pending.total_steps)
                target = self._pending.start_q_rad + fraction * (
                    self._pending.target_q_rad - self._pending.start_q_rad
                )
            # MuJoCo sees the arm reference velocity during contact integration.
            self.data.qvel[self._dof_addresses] = (target - start) / self.timestep_s
            delta = self._commanded_opening - self._setpoint_opening
            speed = self.scene_spec.gripper.opening_speed_m_s
            self._setpoint_opening += float(
                np.clip(delta, -speed * self.timestep_s, speed * self.timestep_s)
            )
            self.data.ctrl[self._actuators] = self._setpoint_opening / 2
            mujoco.mj_step(self.model, self.data)
            self._max_arm_projection_error = max(
                self._max_arm_projection_error,
                float(np.max(np.abs(self.data.qpos[self._qpos_addresses] - target))),
            )
            # Project only six arm joints onto their exact S1 reference. Free
            # object and finger qpos/qvel are never assigned here or in the runner.
            self.data.qpos[self._qpos_addresses] = target
            self.data.qvel[self._dof_addresses] = 0
            self._tick += 1
            self.data.time = self._tick * self.timestep_s
            if (
                self._pending is not None
                and self._pending_step >= self._pending.total_steps
            ):
                self._pending = None
                self._pending_step = 0
            mujoco.mj_forward(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(
                np.isfinite(self.data.qvel)
            ):
                self._abort("safety/validation_abort: nonfinite simulation state")
            if any(w.number for w in self.data.warning):
                self._abort("safety/validation_abort: MuJoCo numerical warning")
            self._update_diagnostics()
        return steps

    def hold(self, duration_s: float) -> int:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("hold duration must be finite and non-negative")
        return self.step(math.ceil(duration_s / self.timestep_s - 1e-12))

    def get_object_pose(self, semantic_name: str) -> Pose:
        """Object body pose in world SI coordinates; actions remain base-relative."""
        self._ensure_open()
        bid = self._object_ids[semantic_name]
        return Pose(
            self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy()
        )

    def get_object_state(self, semantic_name: str) -> dict:
        bid = self._object_ids[semantic_name]
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0
        )
        return {
            "semantic_name": semantic_name,
            "frame": "world",
            **self.get_object_pose(semantic_name).as_dict(),
            "quaternion_wxyz": self.data.xquat[bid].tolist(),
            "linear_velocity_m_s": velocity[3:].tolist(),
            "angular_velocity_rad_s": velocity[:3].tolist(),
            "dynamic": self.scene_spec.object(semantic_name).dynamic,
        }

    def get_gripper_state(self) -> dict:
        self._ensure_open()
        positions = self.data.qpos[self._finger_qpos]
        opening = float(np.sum(positions))
        return {
            "commanded_opening_m": self._commanded_opening,
            "setpoint_opening_m": self._setpoint_opening,
            "measured_opening_m": opening,
            "normalized_gripper": opening / self.scene_spec.gripper.max_opening_m,
            "commanded_normalized_gripper": self._commanded_opening
            / self.scene_spec.gripper.max_opening_m,
            "left_finger_position_m": float(positions[0]),
            "right_finger_position_m": float(positions[1]),
            "state": self._state,
            "bilateral_blue_contact": self._bilateral,
            "finger_contacts": self._finger_contacts,
            "holding": self._holding,
            "slip_distance_m": self._slip,
            "measured_tcp": self.scene_spec.gripper.measured_tcp,
        }

    def contact_records(self) -> list[dict]:
        records = []
        for c in self.data.contact[: self.data.ncon]:
            force = np.zeros(6)
            index = len(records)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            records.append(
                {
                    "geom1": self._geom_names[c.geom1],
                    "geom2": self._geom_names[c.geom2],
                    "distance_m": float(c.dist),
                    "normal_force_n": float(force[0]),
                }
            )
        return records

    def _update_diagnostics(self) -> None:
        contacts = self.contact_records()
        self._last_contacts = contacts
        pairs = set()
        touched = {"left": set(), "right": set()}
        support = False
        for c in contacts:
            names = {c["geom1"], c["geom2"]}
            pair = " / ".join(sorted(names))
            pairs.add(pair)
            self._pair_max_force[pair] = max(
                self._pair_max_force.get(pair, 0), c["normal_force_n"]
            )
            self._max_penetration = max(self._max_penetration, -c["distance_m"])
            if c["normal_force_n"] > self.scene_spec.physics.contact_force_threshold_n:
                for side in touched:
                    if f"{side}_finger_pad" in names:
                        touched[side].update(names - {f"{side}_finger_pad"})
                support |= names == {"blue_block_collision", "red_box_bottom"}
            # Intended base support and object contacts are permitted. Arm
            # proxy contacts and gripper/environment penetration abort episodes.
            arm_contact = any(
                n.endswith("_collision") and n.startswith("Link") for n in names
            )
            tool_environment = any(
                n.startswith("gripper_") or "finger_pad" in n for n in names
            ) and any(n == "work_surface" or n.startswith("red_box_") for n in names)
            if (arm_contact or tool_environment) and c[
                "distance_m"
            ] < -self.scene_spec.physics.collision_abort_penetration_m:
                self._abort(f"collision_abort: {pair}")
        self._pair_counts.update(pairs)
        pair_tuple = tuple(sorted(pairs))
        if pair_tuple != self._last_pairs:
            self.contact_timeline.append(
                {"time_s": self.simulation_time_s, "pairs": list(pair_tuple)}
            )
            self._last_pairs = pair_tuple
        self._finger_contacts = {side: sorted(names) for side, names in touched.items()}
        self._bilateral = all(
            "blue_block_collision" in names for names in touched.values()
        )
        blue = self.get_object_pose("blue_block").position_m
        grasp = self.data.site_xpos[self._grasp_id]
        rotation = self.data.site_xmat[self._grasp_id].reshape(3, 3)
        relative = rotation.T @ (blue - grasp)
        gs, ps = self.scene_spec.gripper, self.scene_spec.physics
        blue_half = np.array(self.scene_spec.object("blue_block").dimensions_m) / 2
        in_region = bool(
            np.all(np.abs(relative) <= blue_half + np.array(gs.finger_half_size_m))
        )
        relative_speed = (
            math.inf
            if self._previous_relative is None
            else float(
                np.linalg.norm(relative - self._previous_relative) / self.timestep_s
            )
        )
        self._previous_relative = relative.copy()
        opening = self.get_gripper_state()["measured_opening_m"]
        opening_command = (
            self._commanded_opening > opening + ps.opening_state_tolerance_m
        )
        lift = float(blue[2] - self._initial_object_positions["blue_block"][2])
        self._max_lift = max(self._max_lift, lift)
        if self._bilateral and in_region and not opening_command:
            if self._candidate_relative is None:
                self._candidate_relative = relative.copy()
                self._candidate_time = 0.0
            self._slip = float(np.linalg.norm(relative - self._candidate_relative))
            self._max_slip = max(self._max_slip, self._slip)
            if (
                relative_speed < ps.settled_linear_speed_m_s
                and self._slip <= ps.max_slip_m
            ):
                self._candidate_time += self.timestep_s
            else:
                self._candidate_time = 0.0
            stable = self._candidate_time >= ps.stable_contact_s
            self._holding = stable and lift >= ps.lift_threshold_m
            self._state = (
                "HOLDING"
                if self._holding
                else "HOLDING_CANDIDATE"
                if stable
                else "CONTACT"
            )
            self._ever_grasped |= self._holding
            self._ever_stable |= self._holding
        else:
            self._candidate_relative = None
            self._candidate_time = 0.0
            self._holding = False
            self._state = (
                "OPENING"
                if opening_command
                else "CLOSING"
                if self._commanded_opening < opening - ps.opening_state_tolerance_m
                else "OPEN"
                if opening > gs.max_opening_m - ps.opening_state_tolerance_m
                else "CLOSED_EMPTY"
            )
        state = self.get_object_state("blue_block")
        settled = (
            support
            and self.inside_target()
            and np.linalg.norm(state["linear_velocity_m_s"])
            <= ps.settled_linear_speed_m_s
            and np.linalg.norm(state["angular_velocity_rad_s"])
            <= ps.settled_angular_speed_rad_s
        )
        self._settled_time = self._settled_time + self.timestep_s if settled else 0.0
        black = self.get_object_pose("black_block").position_m
        self._max_black_displacement = max(
            self._max_black_displacement,
            float(
                np.linalg.norm(black - self._initial_object_positions["black_block"])
            ),
        )

    def object_corners(self, name: str) -> np.ndarray:
        pose = self.get_object_pose(name)
        half = np.array(self.scene_spec.object(name).dimensions_m) / 2
        local = np.array(list(product((-1, 1), repeat=3))) * half
        return local @ pose.rotation_matrix.T + pose.position_m

    def inside_target(self) -> bool:
        target = self.get_object_pose("red_box")
        corners = (
            self.object_corners("blue_block") - target.position_m
        ) @ target.rotation_matrix
        low, high = self.scene_spec.object("red_box").interior_bounds_local_m
        tol = self.scene_spec.physics.placement_tolerance_m
        return bool(
            np.all(corners >= np.array(low) - tol)
            and np.all(corners <= np.array(high) + tol)
        )

    def contact_summary(self) -> dict:
        return {
            "pair_tick_counts": dict(sorted(self._pair_counts.items())),
            "pair_max_normal_force_n": dict(sorted(self._pair_max_force.items())),
            "max_penetration_m": self._max_penetration,
            "max_arm_projection_error_rad": self._max_arm_projection_error,
            "final_contacts": self._last_contacts,
            "fault": self.fault,
        }

    def compute_manipulation_metrics(self) -> dict:
        blue = self.get_object_pose("blue_block")
        black = self.get_object_pose("black_block")
        gs = self.get_gripper_state()
        ps = self.scene_spec.physics
        released = (
            not self._holding
            and not any(
                "blue_block_collision" in names
                for names in self._finger_contacts.values()
            )
            and gs["measured_opening_m"]
            > self.scene_spec.object("blue_block").dimensions_m[0]
            + ps.release_clearance_m
        )
        settled = self._settled_time >= ps.settled_duration_s
        inside = self.inside_target()
        acceptable_black = (
            self._max_black_displacement <= ps.distractor_displacement_limit_m
        )
        return {
            "object_grasped": self._ever_grasped,
            "holding_now": self._holding,
            "lift_height": float(
                blue.position_m[2] - self._initial_object_positions["blue_block"][2]
            ),
            "max_lift_height": self._max_lift,
            "grasp_stable": self._ever_stable,
            "slip_distance": self._max_slip,
            "inside_target": inside,
            "released": released,
            "settled": settled,
            "settled_duration_s": self._settled_time,
            "black_block_displacement": float(
                np.linalg.norm(
                    black.position_m - self._initial_object_positions["black_block"]
                )
            ),
            "max_black_block_displacement": self._max_black_displacement,
            "black_distractor_acceptable": acceptable_black,
            "collision_contact_summary": self.contact_summary(),
            "final_blue_pose": blue.as_dict(),
            "final_black_pose": black.as_dict(),
            "final_gripper_opening": gs["measured_opening_m"],
            "robot_faulted": self.fault is not None,
            "success": bool(
                self._ever_grasped
                and self._ever_stable
                and self._max_slip <= ps.max_slip_m
                and inside
                and settled
                and released
                and acceptable_black
                and self.fault is None
            ),
        }


class ManipulationChunkExecutor(NovaMujocoChunkExecutor):
    """S2 translation contract plus active 0=closed/1=open gripper.

    IK uses a separate frozen-model data instance, so trial qpos do not modify
    the live contact solver, warm starts, object poses, or simulation clock.
    Arbitrary reset-relative orientation remains deferred, exactly as in S2.
    """

    def __init__(self, simulator: ManipulationSimulator):
        super().__init__(simulator, frame=FRAME_NOMINAL_GRASP_CENTER)
        self.kinematics = NovaMujocoSimulator(config=simulator.config)
        self.ik = CartesianIKSolver(self.kinematics)
        self.gripper_records: list[dict] = []

    @staticmethod
    def _action_data(action: Any) -> np.ndarray:
        value = getattr(action, "data", action)
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CartesianTargetError("action must be a numeric SI vector") from exc
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise CartesianTargetError(
                "action must be finite [x,y,z,yaw,pitch,roll,gripper]"
            )
        if np.any(vector[3:6] != 0):
            raise CartesianTargetError(
                "orientation is pinned to reset; yaw, pitch, roll must be 0"
            )
        ManipulationSimulator.validate_gripper(vector[6])
        return vector.copy()

    def execute_chunk(self, chunk: Any) -> ChunkExecutionResult:
        sim = self.simulator
        records = []
        for index, raw_action in enumerate(self._actions(chunk)):
            try:
                if sim.fault:
                    raise ManipulationFault(sim.fault)
                action = self._action_data(raw_action)
                current = sim.get_tool_pose(self.frame)
                distance = float(np.linalg.norm(action[:3] - current.position_m))
                if distance > self.max_cartesian_step_m + 1e-12:
                    raise CartesianTargetError("Cartesian step exceeds 0.02 m")
                self.kinematics.reset(sim.get_joint_position())
                target = Pose(action[:3].copy(), self.reference_rotation.copy())
                solution = self.ik.solve(target, frame=self.frame)
                if not solution.converged:
                    raise IKFailure(f"IK_fail: {solution.status}")
                required = (
                    float(np.max(np.abs(solution.q_rad - sim.get_joint_position())))
                    / sim.config.max_joint_speed_rad_s
                )
                duration = max(
                    required, distance / sim.scene_spec.physics.cartesian_speed_m_s
                )
                sim.command_joint_target(solution.q_rad, duration_s=duration)
                sim.command_gripper(action[6])
                sim.settle()
                records.append(
                    WaypointRecord(
                        index,
                        action,
                        target,
                        solution,
                        sim.get_joint_position(),
                        sim.get_tool_pose(self.frame),
                        tuple(asdict(c) for c in sim.contact_diagnostics()),
                        sim.simulation_time_s,
                    )
                )
                self.gripper_records.append(
                    {"time_s": sim.simulation_time_s, **sim.get_gripper_state()}
                )
            except (ValueError, IKFailure, SimulatorError) as exc:
                return ChunkExecutionResult("rejected", tuple(records), index, str(exc))
        return ChunkExecutionResult("completed", tuple(records))

    def close(self) -> None:
        self.kinematics.close()
