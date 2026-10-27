"""Scene units, composition, provenance, free bodies and nominal tool frames."""

from dataclasses import replace
from itertools import combinations
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from benchmark_scene import ROOT, SCENE_DIR, SCENE_PATH, benchmark_v0, build_scene
from nova_simulator import NovaMujocoSimulator


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_headless_scene_has_semantic_names_dynamic_blocks_and_no_object_weld(scene):
    model, _ = scene
    spec = benchmark_v0()
    for obj in spec.objects:
        body = model.body(obj.semantic_name)
        assert body.id >= 0
        assert int(model.body_dofnum[body.id]) == (6 if obj.dynamic else 0)
        assert obj.source_kind == "temporary_primitive"
        assert "no CAD/STEP" in obj.provenance
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert model.joint(int(model.eq_obj1id[0])).name == "left_finger_slide"
    assert model.joint(int(model.eq_obj2id[0])).name == "right_finger_slide"


@pytest.mark.parametrize(
    ("name", "dimensions", "role", "channel"),
    [
        ("blue_block", (0.060, 0.060, 0.060), "manipulation_target", 2),
        ("black_block", (0.050, 0.050, 0.050), "distractor", None),
        ("red_box", (0.150, 0.150, 0.080), "placement_target", 0),
    ],
)
def test_exact_si_dimensions_colors_roles(scene, name, dimensions, role, channel):
    model, _ = scene
    obj = benchmark_v0().object(name)
    assert obj.dimensions_m == dimensions
    assert obj.role == role
    indices = np.flatnonzero(model.geom_bodyid == model.body(name).id)
    points = []
    for gid in indices:
        assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        color = model.geom_rgba[gid, :3]
        assert np.max(color) < 0.05 if channel is None else color[channel] > 0.8
        points.extend(
            [
                model.geom_pos[gid] - model.geom_size[gid],
                model.geom_pos[gid] + model.geom_size[gid],
            ]
        )
    assert np.ptp(points, axis=0) == pytest.approx(dimensions, abs=1e-14)


def test_open_box_has_only_five_walls_and_unobstructed_interior(scene):
    model, _ = scene
    obj = benchmark_v0().object("red_box")
    low, high = (np.array(v) for v in obj.interior_bounds_local_m)
    assert high - low == pytest.approx([0.140, 0.140, 0.075])
    assert np.all(high - low > benchmark_v0().object("blue_block").dimensions_m)
    gids = np.flatnonzero(model.geom_bodyid == model.body("red_box").id)
    assert {model.geom(int(i)).name for i in gids} == {
        f"red_box_{s}" for s in ("bottom", "left", "right", "front", "rear")
    }
    for gid in gids:
        a, b = (
            model.geom_pos[gid] - model.geom_size[gid],
            model.geom_pos[gid] + model.geom_size[gid],
        )
        # A solid intrusion must overlap the interior with positive volume.
        assert not np.all(np.minimum(b, high) - np.maximum(a, low) > 1e-12)
    # Vertical ray through the opening reaches the floor, not a hidden lid.
    center = np.array(obj.initial_position_world_m)
    from_inside = center + [0, 0, 0.10]
    for gid in gids:
        if model.geom(int(gid)).name != "red_box_bottom":
            local = from_inside[:2] - center[:2] - model.geom_pos[gid, :2]
            assert not np.all(np.abs(local) < model.geom_size[gid, :2])


def test_initial_nonoverlap_and_no_robot_contacts(scene):
    model, data = scene
    objects = benchmark_v0().objects
    for a, b in combinations(objects, 2):
        separation = np.abs(
            np.array(a.initial_position_world_m) - b.initial_position_world_m
        )
        assert np.any(separation >= (np.array(a.dimensions_m) + b.dimensions_m) / 2)
    for c in data.contact[: data.ncon]:
        names = {model.geom(c.geom1).name, model.geom(c.geom2).name}
        assert names <= {
            "work_surface",
            "blue_block_collision",
            "black_block_collision",
        }
        assert c.dist >= -1e-12


@pytest.mark.parametrize("normalized", [0.0, 0.5, 1.0])
def test_nominal_grasp_center_and_symmetric_fingertips(scene, normalized):
    model, data = scene
    gs = benchmark_v0().gripper
    for side in ("left", "right"):
        qadr = int(model.jnt_qposadr[model.joint(f"{side}_finger_slide").id])
        data.qpos[qadr] = normalized * gs.max_opening_m / 2
    mujoco.mj_forward(model, data)
    flange = model.site("tcp").id
    center = model.site("grasp_center").id
    rotation = data.site_xmat[flange].reshape(3, 3)
    assert rotation.T @ (
        data.site_xpos[center] - data.site_xpos[flange]
    ) == pytest.approx([0, 0, 0.197], abs=1e-14)
    tips = [
        rotation.T
        @ (data.site_xpos[model.site(f"{side}_fingertip").id] - data.site_xpos[center])
        for side in ("left", "right")
    ]
    assert tips[0] == pytest.approx([normalized * 0.095 / 2, 0, 0], abs=1e-14)
    assert tips[1] == pytest.approx(-tips[0], abs=1e-14)
    assert gs.measured_tcp is False


def test_arm_frames_limits_and_fk_are_frozen(scene):
    model, data = scene
    frozen = NovaMujocoSimulator()
    q = benchmark_v0().initial_q_rad
    frozen.reset(q)
    for name in ("Link1", "Link2", "Link3", "Link4", "Link5", "Link6"):
        arm_id, old_id = model.body(name).id, frozen.model.body(name).id
        assert model.body_pos[arm_id] == pytest.approx(
            frozen.model.body_pos[old_id], abs=1e-15
        )
        assert model.body_quat[arm_id] == pytest.approx(
            frozen.model.body_quat[old_id], abs=1e-15
        )
    for i in range(1, 7):
        joint, old = model.joint(f"joint{i}").id, frozen.model.joint(f"joint{i}").id
        assert np.array_equal(model.jnt_axis[joint], frozen.model.jnt_axis[old])
        assert np.array_equal(model.jnt_range[joint], frozen.model.jnt_range[old])
    assert data.site_xpos[model.site("tcp").id] == pytest.approx(
        frozen.data.site_xpos[frozen.model.site("tcp").id], abs=1e-14
    )


def test_checked_in_scene_regenerates_and_custom_wall_thickness_loads(tmp_path):
    build_scene(tmp_path)
    for filename in ("arm.xml", "objects.xml", "gripper.xml"):
        assert (tmp_path / filename).read_bytes() == (SCENE_DIR / filename).read_bytes()
    generated = ET.parse(tmp_path / "scene.xml")
    checked = ET.parse(SCENE_PATH)
    generated.find("compiler").set("meshdir", checked.find("compiler").get("meshdir"))
    assert ET.tostring(generated.getroot()) == ET.tostring(checked.getroot())
    custom = benchmark_v0(wall_thickness_m=0.010)
    path = build_scene(tmp_path / "custom", custom)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.geom_size[model.geom("red_box_bottom").id, 2] == 0.005
    assert custom.object("red_box").interior_bounds_local_m[1][0] == pytest.approx(
        0.065
    )


@pytest.mark.parametrize("thickness", [0, -0.001, 0.05, float("nan")])
def test_invalid_container_thickness_rejected(thickness):
    with pytest.raises(ValueError):
        benchmark_v0(wall_thickness_m=thickness)


def test_bad_object_dimensions_rejected():
    with pytest.raises(ValueError):
        replace(benchmark_v0().object("blue_block"), dimensions_m=(0.06, -0.06, 0.06))


def test_original_model_source_still_has_no_added_scene():
    root = ET.parse(ROOT / "nova2.xml").getroot()
    assert root.find(".//body[@name='gripper_mount']") is None
    assert root.find(".//body[@name='blue_block']") is None
