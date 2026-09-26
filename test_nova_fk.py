from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from nova_fk import LEGACY_AXIS_SIGN, load_model, tcp_pose


ROOT = Path(__file__).resolve().parent


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_model_contract() -> None:
    model = load_model()
    assert model.nq == 6
    assert model.nv == 6
    assert model.nmesh == 7
    assert model.nsite == 2
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "nominal_grasp_center") >= 0
    )
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)
    ] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    np.testing.assert_allclose(
        model.jnt_range[:6],
        [
            [-6.28, 6.28],
            [-3.14, 3.14],
            [-2.79, 2.79],
            [-6.28, 6.28],
            [-6.28, 6.28],
            [-6.28, 6.28],
        ],
    )


def test_fk_matches_independent_urdf_transform_product() -> None:
    origins = [
        ((0, 0, 0.2234), (0, 0, 0)),
        ((0, 0, 0), (1.5708, 1.5708, 0)),
        ((-0.28, 0, 0), (0, 0, 0)),
        ((-0.22501, 0, 0.1175), (0, 0, -1.5708)),
        ((0, -0.12, 0), (1.5708, 0, 0)),
        ((0, 0.088004, 0), (-1.5708, 0, 0)),
    ]
    q = np.deg2rad([10, 20, 30, 40, 50, 60])
    transform = np.eye(4)
    transform[:3, 3] = (0, 0, 0.03)
    for value, (xyz, rpy) in zip(q, origins, strict=True):
        origin = np.eye(4)
        origin[:3, :3] = _rz(rpy[2]) @ _ry(rpy[1]) @ _rx(rpy[0])
        origin[:3, 3] = xyz
        joint = np.eye(4)
        joint[:3, :3] = _rz(value)
        transform = transform @ origin @ joint

    pose = tcp_pose(q)
    np.testing.assert_allclose(pose.world_position_m, transform[:3, 3], atol=1e-7)
    np.testing.assert_allclose(pose.rotation_matrix, transform[:3, :3], atol=1e-7)
    np.testing.assert_allclose(
        pose.position_m, transform[:3, 3] - (0, 0, 0.03), atol=1e-7
    )


def test_legacy_axis_hypothesis_is_explicit_q_sign_variant() -> None:
    q = np.deg2rad([10, 20, 30, 40, 50, 60])
    legacy = tcp_pose(q, axis_hypothesis="legacy")
    signed_current = tcp_pose(q * LEGACY_AXIS_SIGN, axis_hypothesis="current")
    np.testing.assert_allclose(legacy.position_m, signed_current.position_m)
    np.testing.assert_allclose(legacy.rotation_matrix, signed_current.rotation_matrix)


def test_visual_assets_are_the_pinned_source_meshes() -> None:
    expected = {
        "Link1.STL": "dfa2d03d4d5ef76a16874af3db8303b8ab642182a22d27856b0d16515b710c79",
        "Link2.STL": "83f56d3e22493104861db5d3ccc75f95f60f74a16b730381a1e6e58bd8fbb6c2",
        "Link3.STL": "a5d536e9807eaee98bee06b504d42e305389b9284494ffbe970ad333b6db2c86",
        "Link4.STL": "c62ef326c5f9b7cc2ae314e4592084a9fd5d312f2783cf28a36ea2f52dae0178",
        "Link5.STL": "fff9c1170d17d237b157ae2f2ea8fdb05febaedbe9072872b9d4f3e83659dcc8",
        "Link6.STL": "872a1c4fa13b45749a482e8c26c2d9144f1f6d8d49dd1109e10c6345f19d197c",
        "base_link.STL": "944a15e41eccba51aada79332f66703b641c9e1f46ecdd5fe9f0f2e775f0cc2d",
    }
    for name, digest in expected.items():
        actual = hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest()
        assert actual == digest, name
