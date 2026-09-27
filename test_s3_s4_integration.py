"""Production S3×S4 integration regressions; frozen phase tests remain unchanged."""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from PIL import Image

from benchmark_scene import ROOT, SCENE_DIR, benchmark_v0
from manipulation_simulator import ManipulationSimulator
from nova_cameras import (
    CAMERA_NAMES,
    DEFAULT_CONFIG,
    NovaMujocoCameraRig,
    _expanded_scene,
)
from run_s3_s4_integration import (
    MONTAGES,
    OBJECT_NAMES,
    PRODUCTION_CAMERA_CONFIG,
    SHAPES,
    STAGES,
    physics_state,
    production_visibility,
    run,
)


@pytest.fixture
def rig():
    with (
        ManipulationSimulator() as sim,
        NovaMujocoCameraRig(sim, config_path=PRODUCTION_CAMERA_CONFIG) as camera,
    ):
        yield camera


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    output = tmp_path_factory.mktemp("integration") / "checkpoint"
    report = run(output)
    observations = json.loads((output / "observation_metadata.json").read_text())[
        "observations"
    ]
    return output, report, observations


def test_production_scene_ownership_and_physics_layout_are_preserved(rig):
    source = rig.simulator.model
    assert source.nq == rig.model.nq == 22
    assert source.nv == rig.model.nv == 20
    assert source.ncam == 0 and rig.model.ncam == 3
    assert rig.model.neq == 1
    assert rig.model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    for field in (
        "qpos0",
        "key_qpos",
        "key_ctrl",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_qposadr",
        "jnt_dofadr",
        "dof_armature",
        "dof_damping",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_solref",
        "geom_solimp",
        "geom_rgba",
        "eq_data",
        "eq_obj1id",
        "eq_obj2id",
        "actuator_gainprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ):
        np.testing.assert_array_equal(
            getattr(rig.model, field), getattr(source, field), err_msg=field
        )
    names = [rig.model.geom(i).name for i in range(rig.model.ngeom)]
    assert not any(name.startswith("s4_") for name in names)
    assert rig.model.ngeom == source.ngeom
    spec = benchmark_v0()
    for name, dims in (
        ("blue_block", (0.060,) * 3),
        ("black_block", (0.050,) * 3),
        ("red_box", (0.150, 0.150, 0.080)),
    ):
        assert spec.object(name).dimensions_m == dims
        bid = rig.model.body(name).id
        gids = np.flatnonzero(rig.model.geom_bodyid == bid)
        bounds = [
            rig.model.geom_pos[i] + sign * rig.model.geom_size[i]
            for i in gids
            for sign in (-1, 1)
        ]
        np.testing.assert_allclose(np.ptp(bounds, axis=0), dims, atol=1e-14, rtol=0)
    assert len([name for name in names if name.startswith("red_box_")]) == 5
    assert spec.object("red_box").wall_thickness_m == 0.005
    assert spec.gripper.max_opening_m == 0.095
    assert spec.gripper.flange_to_grasp_center_m == 0.197
    assert spec.gripper.measured_tcp is False


@pytest.mark.parametrize("name", CAMERA_NAMES)
def test_native_rgb_and_correct_measured_eef_at_reset(rig, name):
    sim = rig.simulator
    obs = rig.observe()
    assert tuple(obs.images) == CAMERA_NAMES
    assert obs.images[name].shape == SHAPES[name]
    assert obs.images[name].dtype == np.uint8
    assert obs.images[name].flags.c_contiguous and not obs.images[name].flags.writeable
    assert rig.controlled_frame == "nominal_grasp_center"
    np.testing.assert_allclose(
        obs.state["eef_state"][:3], sim.get_tool_pose().position_m, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(obs.state["eef_state"][3:], [0, 0, 0, 1])
    assert obs.state_time == 0


def test_default_rig_can_load_modular_production_scene_without_profile_override():
    with ManipulationSimulator() as sim, NovaMujocoCameraRig(sim) as camera:
        assert tuple(camera.observe().images) == CAMERA_NAMES
        assert camera.model.body("blue_block").id >= 0


def test_rendering_leaves_integration_state_contacts_pending_command_and_sources_unchanged(
    rig,
):
    sim = rig.simulator
    files = {p: p.read_bytes() for p in SCENE_DIR.glob("*.xml")}
    target = sim.get_joint_position() + [0, 0.001, 0, 0, 0, 0]
    sim.command_joint_target(target, duration_s=0.1)
    sim.command_gripper(0.5)
    sim.step(5)
    before = physics_state(sim)
    qacc, contacts, metrics = (
        sim.data.qacc.copy(),
        sim.contact_diagnostics(),
        sim.compute_manipulation_metrics(),
    )
    pending, tick, pending_step = sim._pending, sim._tick, sim._pending_step
    rig.observe()
    for name in CAMERA_NAMES:
        rig.render(name)
        rig.camera_pose(name)
        rig.diagnostic_geom_ids(name)
    np.testing.assert_array_equal(physics_state(sim), before)
    np.testing.assert_array_equal(sim.data.qacc, qacc)
    assert contacts == sim.contact_diagnostics()
    assert metrics == sim.compute_manipulation_metrics()
    assert (
        sim._pending is pending
        and sim._tick == tick
        and sim._pending_step == pending_step
    )
    assert files == {p: p.read_bytes() for p in files}


def test_production_pixels_are_owned_by_real_objects_and_rgb_diagnostics_do_not_leak(
    rig,
):
    first = rig.observe()
    report = production_visibility(rig, first)
    generation = rig._generation
    for name in OBJECT_NAMES:
        assert report["front_rgb"][name]["visible_pixels"] > 100
        assert report["front_rgb"][name]["visible_with_expected_color"]
        assert all(
            not g.startswith("s4_") for g in report["front_rgb"][name]["geom_names"]
        )
    assert report["wrist_rgb"]["blue_block"]["visible_pixels"] > 1000
    assert (
        rig._generation == generation
    )  # Validation segmentation is not a policy frame.
    second = rig.observe()
    for name in CAMERA_NAMES:
        np.testing.assert_array_equal(first.images[name], second.images[name])


def test_finger_motion_changes_state_hash_and_observed_aperture(rig):
    sim = rig.simulator
    first = rig.observe()
    sim.command_gripper(0.5)
    sim.hold(1)
    second = rig.observe()
    np.testing.assert_allclose(
        second.state["eef_state"][:3], first.state["eef_state"][:3], atol=1e-12, rtol=0
    )
    assert second.state["eef_state"][6] == pytest.approx(0.5, abs=1e-4)
    assert (
        second.extra["nova_mujoco"]["state_hash"]
        != first.extra["nova_mujoco"]["state_hash"]
    )
    assert (
        second.extra["nova_mujoco"]["frame_generation"]
        == first.extra["nova_mujoco"]["frame_generation"] + 1
    )
    assert (
        np.count_nonzero(first.images["wrist_rgb"] != second.images["wrist_rgb"]) > 100
    )


@pytest.mark.parametrize(
    "failure", ["exception", "none", "wrong_shape", "wrong_dtype", "noncontiguous"]
)
def test_failed_render_never_commits_fresh_observation_or_mutates_physics(
    rig, monkeypatch, failure
):
    previous = rig.observe()
    generation = rig._generation
    previous_pixels = {n: rgb.copy() for n, rgb in previous.images.items()}
    rig.simulator.hold(0.1)
    before = physics_state(rig.simulator)
    draw = rig._draw

    def broken(name):
        if name != "right_rgb":
            return draw(name)
        if failure == "exception":
            raise RuntimeError("injected renderer failure")
        if failure == "none":
            return None
        if failure == "wrong_shape":
            return np.zeros((480, 640), dtype=np.uint8)
        if failure == "wrong_dtype":
            return np.zeros((480, 640, 3), dtype=np.float32)
        return np.zeros((480, 640, 3), dtype=np.uint8)[:, ::-1]

    monkeypatch.setattr(rig, "_draw", broken)
    with pytest.raises(RuntimeError):
        rig.observe()
    assert rig._generation == generation
    np.testing.assert_array_equal(before, physics_state(rig.simulator))
    monkeypatch.setattr(rig, "_draw", draw)
    recovered = rig.observe()
    assert recovered.extra["nova_mujoco"]["frame_generation"] == generation + 1
    assert (
        recovered.extra["nova_mujoco"]["state_hash"]
        != previous.extra["nova_mujoco"]["state_hash"]
    )
    for name in CAMERA_NAMES:
        np.testing.assert_array_equal(previous.images[name], previous_pixels[name])


def test_full_manipulation_with_observations_matches_frozen_physics(evidence):
    _, report, records = evidence
    assert report["status"] == "PASS", report["integration_error"]
    assert report["physics_episode_identical_without_rendering"]
    assert report["manipulation_episode"]["success"]
    assert report["manipulation_episode"]["failure"] is None
    stages = {s["name"]: s for s in report["manipulation_episode"]["stages"]}
    assert all(stages[name]["pass"] for name in STAGES)
    assert stages["grasp"]["gripper"]["bilateral_blue_contact"]
    assert stages["lifted"]["gripper"]["holding"]
    assert stages["transported"]["gripper"]["holding"]
    assert not stages["released"]["gripper"]["holding"]
    metrics = report["manipulation_metrics"]
    assert metrics["inside_target"] and metrics["released"] and metrics["settled"]
    assert metrics["max_lift_height"] > 0.14
    assert metrics["max_black_block_displacement"] < 0.0001
    assert metrics["black_distractor_acceptable"]
    assert not report["s4_test_fixtures_used"]
    assert report["render_failure_probe"]["pass"]
    assert [r["stage"] for r in records] == list(STAGES)


def test_camera_transforms_freshness_and_state_coherence_through_all_phases(evidence):
    _, _, records = evidence
    assert [r["camera_metadata"]["frame_generation"] for r in records] == list(
        range(1, len(STAGES) + 1)
    )
    assert len({r["camera_metadata"]["state_hash"] for r in records}) == len(STAGES)
    assert max(r["wrist_displacement_from_reset_m"] for r in records) > 0.2
    for record in records:
        assert all(record["checks"].values()), record
        assert record["wrist_position_error_m"] < 1e-12
        assert record["wrist_rotation_error"] < 1e-12
        assert (
            record["physics_state_sha256_before"]
            == record["physics_state_sha256_after"]
        )
        assert (
            record["state"]["eef_state"][6] == record["gripper"]["normalized_gripper"]
        )
        for camera in CAMERA_NAMES:
            assert record["shapes"][camera] == list(SHAPES[camera])
            assert record["dtypes"][camera] == "uint8"
        if record["changed_pixel_fraction"]:
            assert all(v > 0 for v in record["changed_pixel_fraction"].values())
    for earlier, later in zip(records, records[1:]):
        assert later["state_time"] > earlier["state_time"]


def test_evidence_contains_native_images_and_never_overwrites(evidence):
    output, report, _ = evidence
    for name in (
        "validation.json",
        "integration_result.json",
        "observation_metadata.json",
        "environment.json",
    ):
        json.loads(
            (output / name).read_text(), parse_constant=lambda value: pytest.fail(value)
        )
    for name in MONTAGES.values():
        assert Image.open(output / name).mode == "RGB"
    for camera, shape in SHAPES.items():
        image = Image.open(output / f"final_scene_{camera}.png")
        assert image.mode == "RGB" and image.size == (shape[1], shape[0])
    assert all(v == 0 for v in report["side_effects"].values())
    assert all(v == 0 for v in report["blocked_io_attempts"].values())
    with pytest.raises(FileExistsError, match="overwrite"):
        run(output)


def test_render_failure_writes_partial_evidence_instead_of_pass(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("injected observation outage")

    monkeypatch.setattr(NovaMujocoCameraRig, "observe", broken)
    report = run(tmp_path / "failed")
    assert report["status"] == "PARTIAL"
    assert report["integration_error"]["category"] == "observation_failure"
    assert report["manipulation_episode"] is None
    records = json.loads((tmp_path / "failed/observation_metadata.json").read_text())
    assert records["observations"] == []
    assert not list((tmp_path / "failed").glob("*.png"))


def test_production_camera_profile_changes_only_uncalibrated_wrist_transform():
    standalone = json.loads(DEFAULT_CONFIG.read_text())
    production = json.loads(PRODUCTION_CAMERA_CONFIG.read_text())
    assert standalone["renderer"] == production["renderer"]
    assert standalone["cameras"][:2] == production["cameras"][:2]
    before, after = standalone["cameras"][2], production["cameras"][2]
    changed = {
        key
        for key in before["parameters"]
        if before["parameters"][key] != after["parameters"][key]
    }
    assert changed == {"position_m", "quaternion_wxyz"}
    assert after["parameters"]["parent"] == "Link6"
    for key in changed:
        assert after["provenance"][key]["status"] == "SIMULATION_ENGINEERING_PARAMETER"
        assert "No measured hand-eye" in after["provenance"][key]["source"]


def test_nested_include_resolution_matches_mujoco_and_preserves_xml(tmp_path):
    (tmp_path / "nested").mkdir()
    main = tmp_path / "scene.xml"
    main.write_text(
        '<mujoco><worldbody><include file="nested/outer.xml"/></worldbody></mujoco>'
    )
    (tmp_path / "nested/outer.xml").write_text(
        '<mujocoinclude><body><include file="inner.xml"/></body></mujocoinclude>'
    )
    (tmp_path / "inner.xml").write_text(
        '<mujocoinclude><geom type="sphere" size=".1"/></mujocoinclude>'
    )
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.xml")}
    source = mujoco.MjModel.from_xml_path(str(main))
    expanded = mujoco.MjModel.from_xml_string(
        ET.tostring(_expanded_scene(main), encoding="unicode")
    )
    np.testing.assert_array_equal(source.geom_size, expanded.geom_size)
    assert original == {p: p.read_bytes() for p in original}


@pytest.mark.parametrize(
    "content",
    [
        '<include file="scene.xml"/>',
        "<include/>",
        '<include file="piece.xml"/><include file="piece.xml"/>',
    ],
)
def test_invalid_duplicate_or_cyclic_includes_rejected(tmp_path, content):
    (tmp_path / "scene.xml").write_text(
        f"<mujoco><worldbody>{content}</worldbody></mujoco>"
    )
    (tmp_path / "piece.xml").write_text(
        '<mujocoinclude><geom type="sphere" size=".1"/></mujocoinclude>'
    )
    with pytest.raises(ValueError, match="include"):
        _expanded_scene(tmp_path / "scene.xml")


def test_no_integration_provider_or_physical_backend_imports():
    import ast

    tree = ast.parse((ROOT / "run_s3_s4_integration.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names.intersection(
        {"openai", "socket", "serial", "cv2", "pyrealsense2", "inspect_robots_agent"}
    )
