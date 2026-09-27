"""benchmark_v0 specifications and reproducible MJCF composition.

All masses, friction, compliance, speeds and simplified gripper dimensions
are SIMULATION ENGINEERING PARAMETERS, never measured hardware parameters.
Object outer dimensions and nominal gripper opening/offset are user supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "scenes" / "benchmark_v0"
SCENE_PATH = SCENE_DIR / "scene.xml"
FROZEN_BASE = "88d93cf686ba0c67afe06b163aa7fb48d65f1a07"
ENGINEERING = "SIMULATION ENGINEERING PARAMETER"


@dataclass(frozen=True)
class BenchmarkObjectSpec:
    semantic_name: str
    role: str
    dimensions_m: tuple[float, float, float]
    initial_position_world_m: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]
    mass_kg: float
    dynamic: bool
    collision_geometry: str
    visual_geometry: str = "primitive_matches_collision"
    source_kind: str = "temporary_primitive"
    initial_quaternion_wxyz: tuple[float, float, float, float] = (1, 0, 0, 0)
    friction: tuple[float, float, float] = (0.9, 0.005, 0.0001)
    wall_thickness_m: float = 0.0
    provenance: str = "S3 operator specification; no CAD/STEP available"
    physics_classification: str = ENGINEERING

    def __post_init__(self) -> None:
        if len(self.dimensions_m) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.dimensions_m
        ):
            raise ValueError("object dimensions must be finite positive metres")
        if len(self.initial_position_world_m) != 3 or not all(
            math.isfinite(v) for v in self.initial_position_world_m
        ):
            raise ValueError("initial object position must contain three finite metres")
        if len(self.initial_quaternion_wxyz) != 4 or not math.isclose(
            sum(v * v for v in self.initial_quaternion_wxyz), 1, abs_tol=1e-9
        ):
            raise ValueError("initial object quaternion must be finite and normalized")
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass must be finite and positive")
        if self.collision_geometry not in {"box", "open_container"}:
            raise ValueError("unsupported collision geometry")
        if self.collision_geometry == "open_container" and not (
            0 < self.wall_thickness_m < min(self.dimensions_m) / 2
        ):
            raise ValueError("wall thickness must leave a nonempty interior")

    @property
    def interior_bounds_local_m(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.collision_geometry != "open_container":
            raise ValueError("only containers have an interior")
        x, y, z = (v / 2 for v in self.dimensions_m)
        t = self.wall_thickness_m
        return (-x + t, -y + t, -z + t), (x - t, y - t, z)


@dataclass(frozen=True)
class GripperSpec:
    max_opening_m: float = 0.095
    flange_to_grasp_center_m: float = 0.197
    measured_tcp: bool = False
    # Flange -> mount is identity; fingers translate along local +/-X.
    finger_half_size_m: tuple[float, float, float] = (0.006, 0.020, 0.025)
    finger_mass_kg: float = 0.08
    opening_speed_m_s: float = 0.06
    position_gain_n_m: float = 2500.0
    finger_damping_ns_m: float = 8.0
    finger_force_limit_n: float = 8.0
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    classification: str = ENGINEERING

    def __post_init__(self) -> None:
        for name in (
            "max_opening_m",
            "flange_to_grasp_center_m",
            "finger_mass_kg",
            "opening_speed_m_s",
            "position_gain_n_m",
            "finger_damping_ns_m",
            "finger_force_limit_n",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.measured_tcp:
            raise ValueError("this simplified nominal gripper has no measured TCP")


@dataclass(frozen=True)
class PhysicsSpec:
    timestep_s: float = 0.002
    gravity_m_s2: tuple[float, float, float] = (0, 0, -9.81)
    contact_solref: tuple[float, float] = (0.008, 1.0)
    contact_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    solver_iterations: int = 100
    # Arm references are projected every tick; large armature makes their
    # within-tick response to contacts negligible. Objects/fingers remain free.
    prescribed_arm_armature: float = 1e6
    max_joint_speed_rad_s: float = 0.30
    cartesian_speed_m_s: float = 0.04
    waypoint_step_m: float = 0.01
    stable_contact_s: float = 0.15
    max_slip_m: float = 0.008
    lift_threshold_m: float = 0.04
    placement_tolerance_m: float = 0.0015
    settled_linear_speed_m_s: float = 0.01
    settled_angular_speed_rad_s: float = 0.10
    settled_duration_s: float = 0.25
    distractor_displacement_limit_m: float = 0.002
    contact_force_threshold_n: float = 0.05
    opening_state_tolerance_m: float = 0.001
    release_clearance_m: float = 0.005
    collision_abort_penetration_m: float = 0.0015
    classification: str = ENGINEERING

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and (
                not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.waypoint_step_m > 0.02:
            raise ValueError("script waypoints must preserve the S2 0.02 m bound")


@dataclass(frozen=True)
class ScriptSpec:
    pregrasp_clearance_m: float = 0.10
    lift_clearance_m: float = 0.15
    release_height_above_floor_m: float = 0.003
    initial_settle_s: float = 0.3
    gripper_settle_s: float = 2.0
    lifted_hold_s: float = 0.5
    transported_hold_s: float = 0.2
    lowered_hold_s: float = 0.2
    final_settle_s: float = 1.0
    classification: str = ENGINEERING


@dataclass(frozen=True)
class BenchmarkSceneSpec:
    objects: tuple[BenchmarkObjectSpec, ...]
    gripper: GripperSpec = GripperSpec()
    physics: PhysicsSpec = PhysicsSpec()
    script: ScriptSpec = ScriptSpec()
    # Base origin stays at world z=.03, as in the frozen model.
    surface_top_world_m: float = 0.03
    initial_q_rad: tuple[float, ...] = (
        0.1057414219467702,
        -0.07307772766776358,
        1.949250880153336,
        -0.30537314570463386,
        -1.5707963268520606,
        1.6765303917661518,
    )
    initial_gripper: float = 1.0
    seed: int | None = None  # No randomness is used.
    name: str = "benchmark_v0"

    def object(self, name: str) -> BenchmarkObjectSpec:
        return next(item for item in self.objects if item.semantic_name == name)

    def as_dict(self) -> dict:
        return asdict(self)


def benchmark_v0(*, wall_thickness_m: float = 0.005) -> BenchmarkSceneSpec:
    return BenchmarkSceneSpec(
        objects=(
            BenchmarkObjectSpec(
                "blue_block",
                "manipulation_target",
                (0.060, 0.060, 0.060),
                (-0.30, -0.15, 0.060),
                (0.025, 0.18, 0.95, 1),
                0.10,
                True,
                "box",
            ),
            BenchmarkObjectSpec(
                "black_block",
                "distractor",
                (0.050, 0.050, 0.050),
                (-0.48, -0.28, 0.055),
                (0.025, 0.025, 0.025, 1),
                0.08,
                True,
                "box",
            ),
            BenchmarkObjectSpec(
                "red_box",
                "placement_target",
                (0.150, 0.150, 0.080),
                (-0.30, 0.08, 0.070),
                (0.85, 0.035, 0.035, 1),
                0.30,
                False,
                "open_container",
                wall_thickness_m=wall_thickness_m,
            ),
        )
    )


def _numbers(values) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def _write_xml(path: Path, node: ET.Element, comment: str) -> None:
    ET.indent(node, space="  ")
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<!-- {comment} -->\n" + ET.tostring(node, encoding="unicode") + "\n"
    )


def build_scene(
    directory: Path = SCENE_DIR, spec: BenchmarkSceneSpec | None = None
) -> Path:
    """Compose a loadable scene, without modifying the frozen nova2.xml.

    Assets/CAD visual replacement belongs here, behind stable semantic bodies.
    Generated files are checked in and tested for byte-identical regeneration.
    """
    spec = spec or benchmark_v0()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(ROOT / "nova2.xml").getroot()
    root.set("model", spec.name)
    root.find("compiler").set("meshdir", os.path.relpath(ROOT / "assets", directory))
    option = root.find("option")
    option.attrib.update(
        gravity=_numbers(spec.physics.gravity_m_s2),
        timestep=str(spec.physics.timestep_s),
        integrator="implicit",
        cone="elliptic",
        iterations=str(spec.physics.solver_iterations),
    )
    defaults = root.find("default/geom")
    defaults.set("solref", _numbers(spec.physics.contact_solref))
    defaults.set("solimp", _numbers(spec.physics.contact_solimp))
    world = root.find("worldbody")
    arm = world.find("body")
    world.remove(arm)
    for joint in arm.iter("joint"):
        joint.set("armature", str(spec.physics.prescribed_arm_armature))
    flange = arm.find(".//body[@name='Link6']")
    ET.SubElement(flange, "include", file="gripper.xml")
    arm_doc = ET.Element("mujocoinclude")
    arm_doc.append(arm)
    ET.SubElement(world, "include", file="arm.xml")
    ET.SubElement(world, "include", file="objects.xml")
    # Static support has its top exactly at the model base mounting plane.
    ET.SubElement(
        world,
        "geom",
        name="work_surface",
        type="box",
        pos=_numbers((-0.20, 0, spec.surface_top_world_m - 0.02)),
        size="0.65 0.50 0.02",
        rgba="0.58 0.58 0.52 1",
        friction="0.9 0.005 0.0001",
    )

    gripper = ET.Element("mujocoinclude")
    mount = ET.SubElement(
        gripper, "body", name="gripper_mount", pos="0 0 0", quat="1 0 0 0"
    )
    ET.SubElement(
        mount,
        "geom",
        name="gripper_stem",
        type="box",
        pos="0 0 0.07",
        size="0.024 0.026 0.055",
        rgba="0.25 0.28 0.32 1",
        mass="0.2",
    )
    ET.SubElement(
        mount,
        "geom",
        name="gripper_palm",
        type="box",
        pos="0 0 0.142",
        size="0.065 0.026 0.012",
        rgba="0.25 0.28 0.32 1",
        mass="0.15",
    )
    gs = spec.gripper
    ET.SubElement(
        mount,
        "site",
        name="grasp_center",
        pos=f"0 0 {gs.flange_to_grasp_center_m}",
        size="0.003",
        rgba="0 1 0 1",
    )
    actuators = ET.SubElement(root, "actuator")
    for side, sign in (("left", 1), ("right", -1)):
        finger = ET.SubElement(
            mount,
            "body",
            name=f"{side}_finger",
            pos=_numbers(
                (sign * gs.finger_half_size_m[0], 0, gs.flange_to_grasp_center_m)
            ),
        )
        ET.SubElement(
            finger,
            "joint",
            name=f"{side}_finger_slide",
            type="slide",
            axis=f"{sign} 0 0",
            range=f"0 {gs.max_opening_m / 2}",
            damping=str(gs.finger_damping_ns_m),
        )
        ET.SubElement(
            finger,
            "geom",
            name=f"{side}_finger_pad",
            type="box",
            size=_numbers(gs.finger_half_size_m),
            mass=str(gs.finger_mass_kg),
            rgba="0.88 0.88 0.84 1",
            friction=_numbers(gs.friction),
            condim="4",
        )
        ET.SubElement(
            finger,
            "site",
            name=f"{side}_fingertip",
            pos=_numbers((-sign * gs.finger_half_size_m[0], 0, 0)),
            size="0.002",
            rgba="0 1 0 1",
        )
        ET.SubElement(
            actuators,
            "position",
            name=f"{side}_finger_drive",
            joint=f"{side}_finger_slide",
            kp=str(gs.position_gain_n_m),
            ctrlrange=f"0 {gs.max_opening_m / 2}",
            forcerange=f"{-gs.finger_force_limit_n} {gs.finger_force_limit_n}",
        )
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        name="symmetric_fingers",
        joint1="left_finger_slide",
        joint2="right_finger_slide",
        polycoef="0 1 0 0 0",
        solref="0.004 1",
    )

    objects = ET.Element("mujocoinclude")
    for obj in spec.objects:
        body = ET.SubElement(
            objects,
            "body",
            name=obj.semantic_name,
            pos=_numbers(obj.initial_position_world_m),
            quat=_numbers(obj.initial_quaternion_wxyz),
        )
        if obj.dynamic:
            ET.SubElement(body, "freejoint", name=f"{obj.semantic_name}_free")
        x, y, z = (v / 2 for v in obj.dimensions_m)
        if obj.collision_geometry == "box":
            pieces = [("collision", (0, 0, 0), (x, y, z))]
        else:
            t = obj.wall_thickness_m
            pieces = [
                ("bottom", (0, 0, -z + t / 2), (x, y, t / 2)),
                ("left", (-x + t / 2, 0, t / 2), (t / 2, y, z - t / 2)),
                ("right", (x - t / 2, 0, t / 2), (t / 2, y, z - t / 2)),
                ("front", (0, -y + t / 2, t / 2), (x - t, t / 2, z - t / 2)),
                ("rear", (0, y - t / 2, t / 2), (x - t, t / 2, z - t / 2)),
            ]
        for name, position, half_size in pieces:
            ET.SubElement(
                body,
                "geom",
                name=f"{obj.semantic_name}_{name}",
                type="box",
                pos=_numbers(position),
                size=_numbers(half_size),
                rgba=_numbers(obj.color_rgba),
                mass=str(obj.mass_kg / len(pieces)),
                friction=_numbers(obj.friction),
                condim="4",
            )
    # Reset keyframe also makes the standalone viewer open at the benchmark start.
    keyframe = ET.SubElement(root, "keyframe")
    qpos = [*spec.initial_q_rad, *(gs.max_opening_m * spec.initial_gripper / 2,) * 2]
    for obj in spec.objects:
        if obj.dynamic:
            qpos.extend((*obj.initial_position_world_m, *obj.initial_quaternion_wxyz))
    ET.SubElement(
        keyframe,
        "key",
        name="benchmark_reset",
        qpos=_numbers(qpos),
        ctrl=_numbers((gs.max_opening_m * spec.initial_gripper / 2,) * 2),
    )
    digest = hashlib.sha256((ROOT / "nova2.xml").read_bytes()).hexdigest()
    _write_xml(
        directory / "arm.xml",
        arm_doc,
        f"Generated frozen arm, source SHA256 {digest}; only armature and gripper include added.",
    )
    _write_xml(
        directory / "gripper.xml",
        gripper,
        "Generated simplified nominal gripper; measured_tcp=false; no USD assembly transform.",
    )
    _write_xml(
        directory / "objects.xml",
        objects,
        "Generated temporary_primitive objects from benchmark_scene.py; physics are engineering parameters.",
    )
    _write_xml(
        directory / "scene.xml",
        root,
        "Generated by benchmark_scene.py; do not hand edit. No camera contract.",
    )
    return directory / "scene.xml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCENE_DIR)
    print(build_scene(parser.parse_args().output))
