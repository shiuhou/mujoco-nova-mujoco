#!/usr/bin/env python3
"""Render the official X-Trainer distal assembly offline.

The generated MJCF is a preview only.  It uses the USD assembly/world transforms
and does not claim that that world frame is a Nova2 flange frame.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import os
from pathlib import Path

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit("Use run_visualize.sh's MuJoCo Python environment") from exc


ROOT = Path(__file__).resolve().parent
PARTS = ("J1_6", "J1_7", "J1_8", "J2_6", "J2_7", "J2_8")


def _quat_from_manifest(matrix: list[list[float]]) -> np.ndarray:
    # USD Gf matrices in the manifest use the row-vector layout. MuJoCo uses
    # column-vector rotations, hence the transpose before conversion.
    rotation = np.asarray(matrix[:3], dtype=float)[:, :3].T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return quat


def _xml(manifest: dict) -> str:
    parts = {part["name"]: part for part in manifest["parts"]}
    assets = "\n".join(
        f'    <mesh name="{name}" file="{(ROOT / "assets/xtrainer_gripper" / (name + "_local.stl")).as_posix()}"/>'
        for name in PARTS
    )
    bodies = []
    for name in PARTS:
        part = parts[name]
        pos = part["assembly_prim_world_translation_m"]
        quat = _quat_from_manifest(part["assembly_prim_world_matrix"])
        bodies.append(
            f'    <body name="{name}" pos="{" ".join(f"{x:.9g}" for x in pos)}" '
            f'quat="{" ".join(f"{x:.9g}" for x in quat)}">\n'
            f'      <geom name="{name}_visual" type="mesh" mesh="{name}" '
            'rgba="0.82 0.84 0.90 1" contype="0" conaffinity="0"/>\n'
            "    </body>"
        )
    return (
        '<mujoco model="xtrainer_gripper_preview">\n'
        '  <compiler meshdir="."/>\n'
        '  <visual><global offwidth="1000" offheight="800"/></visual>\n'
        '  <option gravity="0 0 0"/>'
        "  <asset>\n" + assets + "\n  </asset>\n"
        "  <worldbody>\n"
        '    <light pos="1 -1 1" dir="-1 1 -1" directional="true"/>\n'
        + "\n".join(bodies)
        + "\n  </worldbody>\n</mujoco>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "assets/xtrainer_gripper/manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/xtrainer_gripper_assembly.png")
    )
    parser.add_argument(
        "--xml-output", type=Path, help="also keep the generated preview MJCF"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    xml_text = _xml(manifest)
    if args.xml_output:
        xml_path = args.xml_output
    else:
        fd, temporary_name = tempfile.mkstemp(prefix="xtrainer_gripper_", suffix=".xml")
        os.close(fd)
        xml_path = Path(temporary_name)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=800, width=1000)
    # Set a stable camera through a manually created free camera.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.54, -0.12, 0.36]
    camera.distance = 0.62
    camera.azimuth = 135
    camera.elevation = -20
    renderer.update_scene(data, camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer.render()
    from PIL import Image

    Image.fromarray(renderer.render()).save(args.output)
    print(f"Rendered official X-Trainer gripper preview: {args.output.resolve()}")
    print(f"Generated preview MJCF: {xml_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
