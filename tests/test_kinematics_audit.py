"""Frame, invalid-input and numerical regressions; never access hardware."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cartesian_bridge import CartesianIKSolver, IKConfig, Pose
from nova_simulator import NovaMujocoSimulator


def rotated_simulator():
    sim = NovaMujocoSimulator()
    # Rotate the fixed base; no geometry or joint chain is changed.
    quat = Rotation.from_euler("xyz", [0.4, -0.6, 1.2]).as_quat()
    sim.model.body_quat[sim._base_id] = quat[[3, 0, 1, 2]]
    sim.reset(np.array([0.2, -0.6, 1.4, 0.2, -1.2, -0.5]))
    return sim


def test_rotated_base_jacobian_matches_base_pose_derivative():
    sim = rotated_simulator()
    q = sim.get_joint_position()
    jp, jr = sim.site_jacobian()
    h = 1e-6
    for i in range(6):
        dq = np.eye(6)[i] * h
        sim.set_joint_state_for_kinematics(q + dq)
        plus = sim.get_tool_pose()
        sim.set_joint_state_for_kinematics(q - dq)
        minus = sim.get_tool_pose()
        dp = (plus.position_m - minus.position_m) / (2 * h)
        dr = Rotation.from_matrix(
            plus.rotation_matrix @ minus.rotation_matrix.T
        ).as_rotvec() / (2 * h)
        np.testing.assert_allclose(jp[:, i], dp, atol=1e-8)
        np.testing.assert_allclose(jr[:, i], dr, atol=1e-8)


def test_rotated_base_ik_small_translation():
    sim = rotated_simulator()
    pose = sim.get_tool_pose()
    result = CartesianIKSolver(sim).solve(
        Pose(pose.position_m + [0.005, 0, 0], pose.rotation_matrix)
    )
    assert result.converged


@pytest.mark.parametrize("field", ["max_iterations", "stagnation_iterations"])
@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf"), True])
def test_iteration_counts_require_integers(field, value):
    with pytest.raises(ValueError):
        IKConfig(**{field: value})


@pytest.mark.parametrize("bad", [np.full(6, np.nan), np.full(6, np.inf), np.zeros(5)])
def test_invalid_seed_rejected_without_state_change(bad):
    sim = NovaMujocoSimulator()
    before = sim.get_joint_position()
    with pytest.raises(ValueError):
        CartesianIKSolver(sim).solve(sim.get_tool_pose(), seed_q=bad)
    np.testing.assert_array_equal(before, sim.get_joint_position())


@pytest.mark.parametrize(
    "bad", [np.full((3, 3), np.nan), np.diag([1.0, 1.0, -1.0]), np.zeros((3, 3))]
)
def test_invalid_rotation_rejected(bad):
    sim = NovaMujocoSimulator()
    with pytest.raises(ValueError):
        CartesianIKSolver(sim).solve(Pose(np.zeros(3), bad))


def test_unreachable_is_not_success():
    sim = NovaMujocoSimulator()
    result = CartesianIKSolver(sim).solve(Pose(np.array([10.0, 10.0, 10.0]), np.eye(3)))
    assert not result.converged
    assert np.all(np.isfinite(result.q_rad))
    assert np.all(result.q_rad >= sim.joint_limits[:, 0])
    assert np.all(result.q_rad <= sim.joint_limits[:, 1])


@pytest.mark.parametrize("angle", [0.0, 1e-10, np.pi - 1e-8, np.pi])
def test_rotation_log_against_independent_scipy(angle):
    from cartesian_bridge import rotation_error_vector

    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14)
    matrix = Rotation.from_rotvec(axis * angle).as_matrix()
    result = rotation_error_vector(matrix, np.eye(3))
    np.testing.assert_allclose(
        Rotation.from_rotvec(result).as_matrix(), matrix, atol=1e-8
    )
