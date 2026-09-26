"""Hardware-independent S2 Cartesian and ActionChunk compatibility tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cartesian_bridge import (
    CartesianIKSolver,
    CartesianTargetError,
    IKConfig,
    NovaMujocoChunkExecutor,
    Pose,
    rotation_distance,
)
from nova_simulator import NovaMujocoSimulator


VALIDATED_Q = np.deg2rad([-0.8297, -36.7051, 123.6153, 1.4609, -88.6983, -89.2010])
DIRECTIONS = (
    ("+X", np.array([0.01, 0.0, 0.0])),
    ("-X", np.array([-0.01, 0.0, 0.0])),
    ("+Y", np.array([0.0, 0.01, 0.0])),
    ("-Y", np.array([0.0, -0.01, 0.0])),
    ("+Z", np.array([0.0, 0.0, 0.01])),
    ("-Z", np.array([0.0, 0.0, -0.01])),
)


@dataclass(frozen=True)
class GenericAction:
    data: np.ndarray


@dataclass(frozen=True)
class GenericChunk:
    actions: tuple[GenericAction, ...]


def action(position, *, orientation=(0.0, 0.0, 0.0), gripper=0.0):
    return np.array([*position, *orientation, gripper], dtype=np.float64)


@pytest.mark.parametrize(("name", "delta"), DIRECTIONS)
def test_reachable_10_mm_translations_pin_orientation(name, delta):
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    executor = NovaMujocoChunkExecutor(simulator)

    result = executor.execute_chunk([action(start.position_m + delta)])

    assert result.succeeded, (name, result.reason)
    record = result.records[0]
    achieved_delta = record.achieved_pose.position_m - start.position_m
    assert achieved_delta == pytest.approx(delta, abs=1e-6)
    assert (
        rotation_distance(record.achieved_pose.rotation_matrix, start.rotation_matrix)
        < 1e-6
    )
    assert record.ik.iterations <= executor.ik.config.max_iterations


def test_plus_z_10_mm_is_action_semantics_regression():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    result = NovaMujocoChunkExecutor(simulator).execute_chunk(
        [action(start.position_m + [0.0, 0.0, 0.010])]
    )

    record = result.records[0]
    delta = record.achieved_pose.position_m - start.position_m
    assert result.succeeded
    assert delta[2] == pytest.approx(0.010, abs=1e-6)
    assert abs(delta[0]) < 1e-6
    assert abs(delta[1]) < 1e-6
    assert record.ik.position_error_m < 1e-6
    assert record.ik.orientation_error_rad < 1e-6


def test_repeated_ik_solve_is_deterministic():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    target = Pose(start.position_m + [0.005, -0.005, 0.005], start.rotation_matrix)
    solver = CartesianIKSolver(simulator)

    first = solver.solve(target)
    second = solver.solve(target)
    assert first.status == second.status == "converged"
    assert np.array_equal(first.q_rad, second.q_rad)
    assert first.iterations == second.iterations
    assert first.position_error_m == second.position_error_m


def test_waypoints_execute_in_order_and_final_waypoint_is_reached():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    positions = (
        start.position_m + [0.005, 0.0, 0.0],
        start.position_m + [0.005, 0.005, 0.0],
        start.position_m + [0.005, 0.005, 0.005],
    )
    chunk = GenericChunk(
        tuple(GenericAction(action(position)) for position in positions)
    )

    result = NovaMujocoChunkExecutor(simulator).execute_chunk(chunk)

    assert result.succeeded
    assert [record.index for record in result.records] == [0, 1, 2]
    assert [record.requested_action[:3].tolist() for record in result.records] == [
        position.tolist() for position in positions
    ]
    assert result.records[-1].achieved_pose.position_m == pytest.approx(
        positions[-1], abs=1e-6
    )
    assert all(
        later.simulation_time_s > earlier.simulation_time_s
        for earlier, later in zip(result.records, result.records[1:])
    )


def test_failed_waypoint_stops_remainder():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    waypoints = [
        action(start.position_m + [0.005, 0, 0]),
        action(start.position_m + [0.5, 0, 0]),
        action(start.position_m + [0, 0.005, 0]),
    ]

    result = NovaMujocoChunkExecutor(simulator).execute_chunk(waypoints)

    assert result.status == "rejected"
    assert result.failed_index == 1
    assert len(result.records) == 1
    assert "exceeds" in result.reason


@pytest.mark.parametrize(
    ("bad", "message"),
    (
        ([0.0] * 6, "shape"),
        ([0.0] * 8, "shape"),
        ([0, 0, 0, 0, 0, 0, float("nan")], "finite"),
        ([0, 0, 0, 0, 0, 0, float("inf")], "finite"),
        ([0, 0, 0, 0.01, 0, 0, 0], "orientation is pinned"),
        ([0, 0, 0, 0, 0.01, 0, 0], "orientation is pinned"),
        ([0, 0, 0, 0, 0, 0.01, 0], "orientation is pinned"),
        ([0, 0, 0, 0, 0, 0, 0.01], "gripper is inactive"),
    ),
)
def test_invalid_action_contract_is_rejected(bad, message):
    simulator = NovaMujocoSimulator()
    result = NovaMujocoChunkExecutor(simulator).execute_chunk([bad])
    assert result.status == "rejected"
    assert message in result.reason


def test_xyz_are_metres_and_orientation_slots_are_radians():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    executor = NovaMujocoChunkExecutor(simulator)
    result = executor.execute_chunk([action(start.position_m + [0.005, 0, 0])])
    assert result.succeeded
    assert result.records[0].requested_action[0] - start.position_m[0] == pytest.approx(
        0.005
    )
    rejected = executor.execute_chunk(
        [action(start.position_m, orientation=(0.0, 0.01, 0.0))]
    )
    assert rejected.status == "rejected"


def test_reference_orientation_is_captured_at_reset_not_native_zero():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    reference = simulator.get_link6_pose().rotation_matrix
    assert not np.allclose(reference, np.eye(3))
    executor = NovaMujocoChunkExecutor(simulator)
    start = simulator.get_link6_pose()
    result = executor.execute_chunk([action(start.position_m + [0.0, 0.0, 0.005])])
    assert result.succeeded
    assert (
        rotation_distance(result.records[0].achieved_pose.rotation_matrix, reference)
        < 1e-6
    )


def test_far_unreachable_target_fails_explicitly():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    solver = CartesianIKSolver(simulator)
    result = solver.solve(
        Pose(start.position_m + [3.0, 0.0, 0.0], start.rotation_matrix)
    )
    assert not result.converged
    assert result.status in {"stagnated", "joint_limit", "max_iterations"}


def test_near_workspace_boundary_fails_instead_of_returning_closest_pose():
    q = np.deg2rad([-18.7716, 30.7631, 7.9317, -6.6642, -93.7074, -63.4881])
    simulator = NovaMujocoSimulator()
    simulator.reset(q)
    start = simulator.get_link6_pose()
    result = CartesianIKSolver(simulator).solve(
        Pose(start.position_m + [-0.01, 0.0, 0.0], start.rotation_matrix)
    )
    assert result.status == "stagnated"
    assert result.position_error_m > 1e-3


def test_joint_limit_constrained_failure_is_explicit():
    simulator = NovaMujocoSimulator()
    limits = simulator.joint_limits
    q = np.zeros(6)
    q[2] = limits[2, 1]
    simulator.reset(q)
    start = simulator.get_link6_pose()
    result = CartesianIKSolver(simulator).solve(
        Pose(start.position_m + [0.0, 0.02, 0.0], start.rotation_matrix)
    )
    assert not result.converged
    assert result.status in {"joint_limit", "stagnated", "max_iterations"}


def test_bounded_iteration_count():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    start = simulator.get_link6_pose()
    solver = CartesianIKSolver(simulator, config=IKConfig(max_iterations=1))
    result = solver.solve(
        Pose(start.position_m + [0.01, 0.0, 0.0], start.rotation_matrix)
    )
    assert result.status == "max_iterations"
    assert result.iterations == 1


def test_invalid_orientation_matrix_rejected():
    simulator = NovaMujocoSimulator()
    start = simulator.get_link6_pose()
    with pytest.raises(CartesianTargetError, match="proper"):
        CartesianIKSolver(simulator).solve(Pose(start.position_m, np.zeros((3, 3))))
