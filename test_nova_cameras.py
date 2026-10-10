"""S4 observations are tested without devices, providers, or a display server."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from nova_cameras import (
    CAMERA_NAMES,
    DEFAULT_CONFIG,
    ENGINEERING,
    VERIFIED,
    TEST_FIXTURES,
    NovaMujocoCameraRig,
    RendererConfig,
    load_camera_config,
    relative_orientation,
)
from nova_simulator import NovaMujocoSimulator
from s4_validation import fixture_report, observation_repeat_report, rgb_repeat_report

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def rig():
    samples = json.loads((ROOT / "real_nova2_samples.json").read_text())["samples"]
    with NovaMujocoSimulator() as simulator:
        simulator.reset(np.deg2rad(samples[0]["q_deg"]))
        with NovaMujocoCameraRig(simulator, fixture_path=TEST_FIXTURES) as camera_rig:
            yield camera_rig


def test_logical_contract_and_coherent_three_view_observation(rig):
    q_before = rig.simulator.get_joint_position()
    obs = rig.observe(instruction="Scripted fixture observation")
    assert tuple(obs.images) == tuple(obs.image_times) == CAMERA_NAMES
    assert "top_rgb" not in obs.images
    assert obs.instruction == "Scripted fixture observation"
    for name, shape in zip(
        CAMERA_NAMES, ((480, 640, 3), (480, 640, 3), (720, 1280, 3)), strict=True
    ):
        assert obs.images[name].shape == shape
        assert obs.images[name].dtype == np.uint8
        assert obs.images[name].flags.c_contiguous
        assert not obs.images[name].flags.writeable
    np.testing.assert_array_equal(obs.state["joint_pos"], q_before)
    np.testing.assert_allclose(
        obs.state["eef_state"][:3], rig.simulator.get_link6_pose().position_m
    )
    np.testing.assert_array_equal(obs.state["eef_state"][3:], np.zeros(4))
    assert obs.state_time == rig.simulator.simulation_time_s == 0
    np.testing.assert_array_equal(rig.simulator.get_joint_position(), q_before)
    meta = obs.extra["nova_mujoco"]
    assert meta["camera"]["same_state_snapshot"]
    for frame in meta["camera"]["frames"].values():
        assert frame["state_time"] == frame["sim_time"] == obs.state_time
        assert frame["state_hash"] == meta["state_hash"]
        assert frame["frame_generation"] == meta["frame_generation"]
        assert (
            frame["render_finished_host_monotonic_s"]
            >= frame["render_started_host_monotonic_s"]
        )
        assert not frame["physical_exposure"]


def test_static_observation_repeats_and_generation_always_advances(rig):
    first = rig.observe()
    second = rig.observe()
    assert observation_repeat_report(first, second)["passed"]
    assert (
        second.extra["nova_mujoco"]["frame_generation"]
        == first.extra["nova_mujoco"]["frame_generation"] + 1
    )
    assert (
        first.extra["nova_mujoco"]["state_hash"]
        == second.extra["nova_mujoco"]["state_hash"]
    )


def test_fixed_and_wrist_rigid_transforms_under_movement(rig):
    poses = {name: rig.camera_pose(name) for name in CAMERA_NAMES}
    first = rig.observe()
    target = rig.simulator.get_joint_position() + [0.12, -0.04, 0.03, 0, 0, 0.08]
    rig.simulator.command_joint_target(target)
    rig.simulator.settle()
    second = rig.observe()
    for name in CAMERA_NAMES[:2]:
        np.testing.assert_array_equal(rig.camera_pose(name)[0], poses[name][0])
        np.testing.assert_array_equal(rig.camera_pose(name)[1], poses[name][1])
    p, r = rig.camera_pose("wrist_rgb")
    assert np.linalg.norm(p - poses["wrist_rgb"][0]) > 0.01
    link = mujoco.mj_name2id(rig.simulator.model, mujoco.mjtObj.mjOBJ_BODY, "Link6")
    parent_p = rig.simulator.data.xpos[link]
    parent_r = rig.simulator.data.xmat[link].reshape(3, 3)
    spec = rig.specs["wrist_rgb"]
    local_r = np.empty(9)
    mujoco.mju_quat2Mat(local_r, np.array(spec.quaternion_wxyz))
    np.testing.assert_allclose(p, parent_p + parent_r @ spec.position_m, atol=1e-12)
    np.testing.assert_allclose(r, parent_r @ local_r.reshape(3, 3), atol=1e-12)
    for name in CAMERA_NAMES:
        assert np.count_nonzero(first.images[name] != second.images[name]) > 100
    assert second.state_time > first.state_time
    assert (
        second.extra["nova_mujoco"]["state_hash"]
        != first.extra["nova_mujoco"]["state_hash"]
    )


def test_state_change_without_time_advance_and_reset_generation(rig):
    first = rig.observe()
    q = rig.simulator.get_joint_position()
    rig.simulator.set_joint_state_for_kinematics(q + [0.1, 0, 0, 0, 0, 0])
    second = rig.observe()
    assert first.state_time == second.state_time == 0
    assert (
        first.extra["nova_mujoco"]["state_hash"]
        != second.extra["nova_mujoco"]["state_hash"]
    )
    rig.simulator.reset(q)
    third = rig.observe()
    assert (
        third.extra["nova_mujoco"]["frame_generation"]
        > second.extra["nova_mujoco"]["frame_generation"]
    )
    assert observation_repeat_report(first, third)["passed"]


def test_roles_projections_rgb_colors_and_distinct_views(rig):
    obs = rig.observe()
    report = fixture_report(rig, obs)
    assert all(
        entry["inside_frustum"] and entry["color_visible"]
        for camera in report.values()
        for entry in camera.values()
    ), report
    # Projection/color agreement also catches BGR channels and vertical flipping.
    assert (
        np.mean(np.abs(obs.images["front_rgb"].astype(float) - obs.images["right_rgb"]))
        > 5
    )
    for i, name in enumerate(CAMERA_NAMES):
        for other in CAMERA_NAMES[i + 1 :]:
            a = np.array(report[name]["s4_red_target"]["pixel_uv"]) / [
                rig.specs[name].width,
                rig.specs[name].height,
            ]
            b = np.array(report[other]["s4_red_target"]["pixel_uv"]) / [
                rig.specs[other].width,
                rig.specs[other].height,
            ]
            assert np.linalg.norm(a - b) > 0.03


@pytest.mark.parametrize("name", CAMERA_NAMES)
def test_independent_render_matches_observation(rig, name):
    frame = rig.render(name)
    obs = rig.observe()
    assert rgb_repeat_report(frame.rgb, obs.images[name])["passed"]
    assert frame.metadata["state_hash"] == obs.extra["nova_mujoco"]["state_hash"]
    assert (
        frame.metadata["frame_generation"]
        < obs.extra["nova_mujoco"]["frame_generation"]
    )


def test_partial_render_failure_returns_no_observation(rig, monkeypatch):
    before = rig.observe().extra["nova_mujoco"]["frame_generation"]
    draw = rig._draw

    def failing(name):
        if name == "right_rgb":
            raise RuntimeError("injected render failure")
        return draw(name)

    monkeypatch.setattr(rig, "_draw", failing)
    with pytest.raises(RuntimeError, match="injected"):
        rig.observe()
    monkeypatch.setattr(rig, "_draw", draw)
    assert rig.observe().extra["nova_mujoco"]["frame_generation"] == before + 1


def test_render_model_preserves_arm_model_arrays_and_source_file(rig):
    source = rig.simulator.model
    for field in (
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
    ):
        np.testing.assert_array_equal(getattr(rig.model, field), getattr(source, field))
    assert source.ncam == 0
    assert rig.model.ncam == 3
    assert source.nq == rig.model.nq == 6
    assert ET.parse(ROOT / "nova2.xml").getroot().find(".//camera") is None


def test_debug_frustums_are_optional_offline_output(rig):
    before = rig.observe()
    assert len(rig.frustum_lines("wrist_rgb")) == 8
    debug = rig.render_debug()
    assert debug.shape == (720, 1280, 3)
    after = rig.observe()
    assert observation_repeat_report(before, after)["passed"]


def test_additional_scene_object_state_is_preserved(tmp_path):
    # A future scene owner adds object physics; S4 copies the complete qpos layout.
    root = ET.parse(ROOT / "nova2.xml").getroot()
    root.find("compiler").set("meshdir", str(ROOT / "assets"))
    obj = ET.SubElement(
        root.find("worldbody"), "body", name="future_object", pos="-.2 -.1 .2"
    )
    ET.SubElement(obj, "freejoint", name="future_object_joint")
    ET.SubElement(
        obj, "geom", type="box", size=".03 .03 .03", rgba=".1 .8 .1 1", group="3"
    )
    path = tmp_path / "scene.xml"
    ET.ElementTree(root).write(path)
    with NovaMujocoSimulator(path) as simulator:
        simulator.data.qpos[-7:-4] = [-0.3, -0.2, 0.3]
        mujoco.mj_forward(simulator.model, simulator.data)
        with NovaMujocoCameraRig(simulator) as rig:
            first = rig.render("front_rgb")
            np.testing.assert_array_equal(rig.data.qpos, simulator.data.qpos)
            simulator.data.qpos[-7] += 0.15
            mujoco.mj_forward(simulator.model, simulator.data)
            second = rig.render("front_rgb")
            assert not np.array_equal(first.rgb, second.rgb)


def test_configuration_provenance_and_explicit_wrist_transform():
    specs, renderer = load_camera_config()
    assert tuple(s.name for s in specs) == CAMERA_NAMES
    document = json.loads(DEFAULT_CONFIG.read_text())
    for entry in document["cameras"]:
        assert entry["provenance"]["width"]["status"] == VERIFIED
        for key in ("position_m", "quaternion_wxyz", "fovy_deg", "parent"):
            assert entry["provenance"][key]["status"] == ENGINEERING
    assert renderer.near_m == 0.01 and renderer.far_m == 10


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "top_rgb"},
        {"parent": "base_link"},
        {"position_m": (float("nan"), 0, 0)},
        {"quaternion_wxyz": (1, 0, 0, 1)},
        {"width": -1},
        {"height": 1.5},
        {"fovy_deg": float("inf")},
        {"fovy_deg": 180},
    ],
)
def test_bad_camera_configuration_rejected(changes):
    spec = load_camera_config()[0][0]
    with pytest.raises(ValueError):
        replace(spec, **changes)


@pytest.mark.parametrize(
    "values", [{"near_m": 0}, {"far_m": 0.001}, {"multisamples": 3}]
)
def test_bad_renderer_configuration_rejected(values):
    with pytest.raises(ValueError):
        RendererConfig(**values)


def test_reordered_config_returns_canonical_key_order(tmp_path):
    doc = json.loads(DEFAULT_CONFIG.read_text())
    doc["cameras"].reverse()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    with (
        NovaMujocoSimulator() as sim,
        NovaMujocoCameraRig(sim, config_path=path) as rig,
    ):
        assert tuple(rig.observe().images) == CAMERA_NAMES


def test_relative_orientation_convention_is_radians():
    yaw, pitch, roll = 0.2, -0.1, 0.3

    def r(axis, angle):
        v = np.zeros(3)
        v[axis] = 1
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

    ref = r(0, 0.7)
    actual = r(2, yaw) @ r(1, -pitch) @ r(0, roll) @ ref
    np.testing.assert_allclose(
        relative_orientation(actual, ref), [yaw, pitch, roll], atol=1e-14
    )
    with pytest.raises(ValueError, match="singularity"):
        relative_orientation(r(1, np.pi / 2), np.eye(3))


def test_fresh_rig_reproduces_static_image(rig):
    first = rig.observe()
    with NovaMujocoCameraRig(rig.simulator, fixture_path=TEST_FIXTURES) as independent:
        second = independent.observe()
        for name in CAMERA_NAMES:
            assert rgb_repeat_report(first.images[name], second.images[name])["passed"]
            assert (
                first.extra["nova_mujoco"]["state_hash"]
                == second.extra["nova_mujoco"]["state_hash"]
            )
            np.testing.assert_array_equal(
                rig.camera_pose(name)[0], independent.camera_pose(name)[0]
            )
            np.testing.assert_array_equal(
                rig.camera_pose(name)[1], independent.camera_pose(name)[1]
            )


@pytest.mark.parametrize(
    "change", ["duplicate_name", "missing_source", "renderer_source"]
)
def test_incomplete_config_provenance_or_role_set_rejected(tmp_path, change):
    doc = json.loads(DEFAULT_CONFIG.read_text())
    if change == "duplicate_name":
        doc["cameras"][1] = doc["cameras"][0]
    elif change == "missing_source":
        del doc["cameras"][0]["provenance"]["position_m"]
    else:
        doc["renderer"]["status"] = VERIFIED
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_camera_config(path)


def test_closed_rig_and_invalid_name_fail_explicitly(rig):
    with pytest.raises(ValueError, match="unknown camera"):
        rig.render("front")
    rig.close()
    rig.close()
    with pytest.raises(RuntimeError, match="closed"):
        rig.observe()
    assert rig.simulator.get_joint_position().shape == (6,)


def test_no_physical_backend_imports():
    forbidden = {
        "cv2",
        "serial",
        "socket",
        "openai",
        "inspect_robots",
        "inspect_robots_agent",
    }
    tree = ast.parse((ROOT / "nova_cameras.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(x.name.split(".")[0] not in forbidden for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


def test_audit_guard_blocks_physical_camera_serial_and_network():
    import socket

    with pytest.raises(RuntimeError, match="network"):
        socket.socket()
    for path in ("/dev/video0", "/dev/v4l/by-id/nonexistent", "/dev/ttyUSB0"):
        with pytest.raises(RuntimeError, match="physical"):
            open(path, "rb")
