#!/usr/bin/env python3
"""Build a separate Nova2 + calibrated gripper preview without editing nova2.xml."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINGER_ASSETS = ROOT / "assets/xtrainer_gripper"
MANIFEST = FINGER_ASSETS / "manifest.json"


def _relative_part_transform(
    parent_name: str, child_name: str
) -> tuple[list[float], list[float]]:
    """Return child pose in the parent part frame from the USD export manifest.

    The manifest stores USD matrices with translation in the final row.  The
    exported STL remains prim-local, so this relative transform belongs on the
    MuJoCo body that owns that STL.
    """
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parts = {part["name"]: part for part in data["parts"]}
    parent = parts[parent_name]["assembly_prim_world_matrix"]
    child = parts[child_name]["assembly_prim_world_matrix"]
    rp = [row[:3] for row in parent[:3]]
    rc = [row[:3] for row in child[:3]]
    tp = parent[3][:3]
    tc = child[3][:3]
    # R_rel = R_parent.T @ R_child; t_rel = R_parent.T @ (t_child-t_parent)
    r = [
        [sum(rp[k][i] * rc[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]
    delta = [tc[i] - tp[i] for i in range(3)]
    t = [sum(rp[k][i] * delta[k] for k in range(3)) for i in range(3)]
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    else:
        # These nearly quarter-turn matrices have a positive trace in the
        # supplied asset; retain a complete fallback for future exports.
        i = max(range(3), key=lambda n: r[n][n])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-16, 1 + r[i][i] - r[j][j] - r[k][k])) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i + 1] = 0.25 * s
        q[0] = (r[k][j] - r[j][k]) / s
        q[j + 1] = (r[j][i] + r[i][j]) / s
        q[k + 1] = (r[k][i] + r[i][k]) / s
    return t, q


def build(output: Path) -> None:
    source = ET.parse(ROOT / "nova2.xml")
    root = source.getroot()
    asset = root.find("asset")
    link6 = root.find(".//body[@name='Link6']")
    if asset is None or link6 is None:
        raise RuntimeError("nova2.xml is missing asset or Link6")

    # Keep official visual meshes in their extracted prim-local coordinates.
    for name in ("J1_6", "J1_7", "J1_8"):
        ET.SubElement(
            asset,
            "mesh",
            name=f"xtrainer_{name}",
            file=f"xtrainer_gripper/{name}_local.stl",
        )

    mount = ET.SubElement(link6, "body", name="xtrainer_gripper_mount", pos="0 0 0")
    ET.SubElement(
        mount,
        "geom",
        name="xtrainer_J1_6_visual",
        type="mesh",
        mesh="xtrainer_J1_6",
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.82 0.84 0.90 1",
    )
    ET.SubElement(
        mount,
        "geom",
        name="xtrainer_body_collision",
        type="box",
        pos="0 0 0.02525",
        size="0.08 0.068 0.06125",
        group="1",
    )
    left_pos, left_quat = _relative_part_transform("J1_6", "J1_7")
    right_pos, right_quat = _relative_part_transform("J1_6", "J1_8")
    left_pos_s = " ".join(f"{v:.9g}" for v in left_pos)
    right_pos_s = " ".join(f"{v:.9g}" for v in right_pos)
    left_quat_s = " ".join(f"{v:.9g}" for v in left_quat)
    right_quat_s = " ".join(f"{v:.9g}" for v in right_quat)
    # The USD defines a local-Z prismatic axis (world +/-X after the official
    # finger rotations). Keep its authored limits here. The physical 95->0 mm
    # opening is a calibration constraint, but cannot be imposed on these
    # prim-local meshes without changing their official relative placement.
    left = ET.SubElement(
        mount, "body", name="xtrainer_left_finger", pos=left_pos_s, quat=left_quat_s
    )
    ET.SubElement(
        left,
        "joint",
        name="gripper_left_slide",
        type="slide",
        axis="0 0 1",
        range="0 0.04",
        damping="0",
    )
    ET.SubElement(
        left,
        "geom",
        name="xtrainer_J1_7_visual",
        type="mesh",
        mesh="xtrainer_J1_7",
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.82 0.84 0.90 1",
    )
    ET.SubElement(
        left,
        "geom",
        name="xtrainer_left_collision",
        type="box",
        pos="0.024 0 0.0615",
        size="0.024 0.013 0.0615",
        group="1",
    )
    right = ET.SubElement(
        mount, "body", name="xtrainer_right_finger", pos=right_pos_s, quat=right_quat_s
    )
    ET.SubElement(
        right,
        "joint",
        name="gripper_right_slide",
        type="slide",
        axis="0 0 1",
        range="0 0.04",
        damping="0",
    )
    ET.SubElement(
        right,
        "geom",
        name="xtrainer_J1_8_visual",
        type="mesh",
        mesh="xtrainer_J1_8",
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.82 0.84 0.90 1",
    )
    ET.SubElement(
        right,
        "geom",
        name="xtrainer_right_collision",
        type="box",
        pos="-0.024 0 0.0615",
        size="0.024 0.013 0.0615",
        group="1",
    )
    ET.SubElement(
        mount,
        "site",
        name="grasp_center",
        pos="0 0 0.197",
        size="0.004",
        rgba="0.1 0.9 0.2 1",
    )
    ET.SubElement(
        mount,
        "site",
        name="gripper_base",
        pos="0 0 0",
        size="0.003",
        rgba="0.9 0.6 0.1 1",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(source, space="  ")
    source.write(output, encoding="utf-8", xml_declaration=True)


def write_gripper_fragment(path: Path) -> None:
    left_pos, left_quat = _relative_part_transform("J1_6", "J1_7")
    right_pos, right_quat = _relative_part_transform("J1_6", "J1_8")
    left_pos_s = " ".join(f"{v:.9g}" for v in left_pos)
    right_pos_s = " ".join(f"{v:.9g}" for v in right_pos)
    left_quat_s = " ".join(f"{v:.9g}" for v in left_quat)
    right_quat_s = " ".join(f"{v:.9g}" for v in right_quat)
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Standalone reusable gripper preview. Mounting transform is supplied by the parent. -->
<mujoco model="xtrainer_gripper">
  <compiler meshdir="." angle="radian"/>
  <asset>
    <mesh name="xtrainer_J1_6" file="assets/xtrainer_gripper/J1_6_local.stl"/>
    <mesh name="xtrainer_J1_7" file="assets/xtrainer_gripper/J1_7_local.stl"/>
    <mesh name="xtrainer_J1_8" file="assets/xtrainer_gripper/J1_8_local.stl"/>
  </asset>
  <worldbody>
    <body name="xtrainer_gripper_mount">
      <geom name="xtrainer_J1_6_visual" type="mesh" mesh="xtrainer_J1_6" contype="0" conaffinity="0"/>
      <body name="xtrainer_left_finger" pos="{left_pos_s}" quat="{left_quat_s}">
        <joint name="gripper_left_slide" type="slide" axis="0 0 1" range="0 0.04"/>
        <geom name="xtrainer_J1_7_visual" type="mesh" mesh="xtrainer_J1_7" contype="0" conaffinity="0"/>
      </body>
      <body name="xtrainer_right_finger" pos="{right_pos_s}" quat="{right_quat_s}">
        <joint name="gripper_right_slide" type="slide" axis="0 0 1" range="0 0.04"/>
        <geom name="xtrainer_J1_8_visual" type="mesh" mesh="xtrainer_J1_8" contype="0" conaffinity="0"/>
      </body>
      <site name="grasp_center" pos="0 0 0.197" size="0.004"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(text, encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "nova2_gripper.xml")
    parser.add_argument("--gripper-fragment", type=Path, default=ROOT / "gripper.xml")
    args = parser.parse_args()
    build(args.output.resolve())
    write_gripper_fragment(args.gripper_fragment.resolve())
    print(f"Wrote composed model: {args.output.resolve()}")
    print(f"Wrote integration notes fragment: {args.gripper_fragment.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
