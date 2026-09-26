"""Hardware-independent S1 deterministic Nova2 execution tests."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nova_fk import tcp_pose
from nova_simulator import (
    FRAME_NOMINAL_GRASP_CENTER,
    JointTargetError,
    NovaMujocoSimulator,
    SimulatorConfig,
)


VALIDATED_Q_DEG = np.array([-0.8297, -36.7051, 123.6153, 1.4609, -88.6983, -89.2010])
VALIDATED_Q = np.deg2rad(VALIDATED_Q_DEG)


def test_model_load_reset_and_link6_pose_match_existing_fk():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)

    assert np.array_equal(simulator.get_joint_position(), VALIDATED_Q)
    pose = simulator.get_link6_pose()
    expected = tcp_pose(VALIDATED_Q)
    assert np.allclose(pose.position_m, expected.position_m, atol=1e-12, rtol=0)
    assert np.allclose(
        pose.rotation_matrix, expected.rotation_matrix, atol=1e-12, rtol=0
    )


def test_nominal_grasp_center_is_explicit_plus_z_197_mm():
    simulator = NovaMujocoSimulator()
    simulator.reset(VALIDATED_Q)
    flange = simulator.get_link6_pose()
    grasp = simulator.get_tool_pose(FRAME_NOMINAL_GRASP_CENTER)

    expected = flange.position_m + flange.rotation_matrix @ np.array([0.0, 0.0, 0.197])
    assert np.allclose(grasp.position_m, expected, atol=1e-12, rtol=0)
    assert np.allclose(
        grasp.rotation_matrix, flange.rotation_matrix, atol=1e-12, rtol=0
    )


def test_deterministic_reset_and_fixed_timestep_joint_execution():
    config = SimulatorConfig(timestep_s=0.002, max_joint_speed_rad_s=0.5)
    first = NovaMujocoSimulator(config=config)
    second = NovaMujocoSimulator(config=config)
    first.reset(VALIDATED_Q)
    second.reset(VALIDATED_Q)
    target = VALIDATED_Q + np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])

    command_a = first.command_joint_target(target)
    command_b = second.command_joint_target(target)
    assert command_a.total_steps == command_b.total_steps == 30
    assert first.settle() == second.settle() == 30
    assert np.array_equal(first.get_joint_position(), second.get_joint_position())
    assert np.allclose(first.get_joint_position(), target, atol=1e-15, rtol=0)
    assert first.simulation_time_s == pytest.approx(0.06)
    assert second.simulation_time_s == pytest.approx(0.06)

    first.reset(target)
    assert first.simulation_time_s == 0
    assert np.array_equal(first.get_joint_position(), target)


def test_step_does_not_require_wall_clock_and_zero_steps_is_noop():
    simulator = NovaMujocoSimulator()
    assert simulator.step(0) == 0
    assert simulator.simulation_time_s == 0
    assert simulator.step(5) == 5
    assert simulator.simulation_time_s == pytest.approx(5 * simulator.timestep_s)


@pytest.mark.parametrize("joint_index", range(6))
def test_each_exact_joint_limit_is_accepted(joint_index):
    simulator = NovaMujocoSimulator()
    limits = simulator.joint_limits
    for side in (0, 1):
        q = np.zeros(6)
        q[joint_index] = limits[joint_index, side]
        simulator.reset(q)
        assert simulator.get_joint_position()[joint_index] == limits[joint_index, side]


@pytest.mark.parametrize("joint_index", range(6))
def test_each_joint_just_outside_limit_is_rejected(joint_index):
    simulator = NovaMujocoSimulator()
    limits = simulator.joint_limits
    for value in (limits[joint_index, 0] - 1e-9, limits[joint_index, 1] + 1e-9):
        q = np.zeros(6)
        q[joint_index] = value
        with pytest.raises(JointTargetError, match=f"joint{joint_index + 1}"):
            simulator.reset(q)


@pytest.mark.parametrize(
    "bad",
    (
        [0.0] * 5,
        [0.0] * 7,
        [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
        [0.0, 0.0, 0.0, 0.0, 0.0, float("inf")],
    ),
)
def test_invalid_joint_targets_are_rejected(bad):
    with pytest.raises(JointTargetError):
        NovaMujocoSimulator().reset(bad)


def test_joint_command_rejects_impossible_duration_instead_of_clamping():
    simulator = NovaMujocoSimulator(config=SimulatorConfig(max_joint_speed_rad_s=1.0))
    with pytest.raises(JointTargetError, match="speed limit"):
        simulator.command_joint_target([0.1, 0, 0, 0, 0, 0], duration_s=0.01)


def test_state_contract_has_future_camera_names_but_no_camera_access():
    simulator = NovaMujocoSimulator()
    state = simulator.get_robot_state()
    assert state["joint_pos"].shape == (6,)
    assert state["eef_position_m"].shape == (3,)
    assert state["logical_camera_names"] == ("front_rgb", "right_rgb", "wrist_rgb")


def test_runtime_modules_do_not_import_network_serial_camera_or_openai():
    prohibited = {"socket", "serial", "cv2", "openai"}
    for filename in ("nova_simulator.py", "cartesian_bridge.py"):
        tree = ast.parse(Path(filename).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported.isdisjoint(prohibited)
