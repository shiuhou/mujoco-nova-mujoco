#!/usr/bin/env python3
"""Extract the official X-Trainer finger meshes from a USD stage.

The exported meshes stay in their USD prim-local coordinates.  The manifest
records the original prim paths and assembly transforms so a later Nova2 tool
attachment can apply a measured flange-to-tool transform explicitly.

Requires Pixar USD Python bindings (``python -m pip install usd-core``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Iterable

from pxr import Gf, Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = (
    REPO_ROOT / "assets/xtrainer_official/X-Trainer-URDF/urdf/"
    "x-trainer_asm-0226.SLDASM/configuration/"
    "x-trainer_asm-0226.SLDASM_base.usd"
)
DEFAULT_OUTPUT = REPO_ROOT / "assets/xtrainer_gripper"
DEFAULT_PRIMS = ("J1_7", "J1_8")
EXTRACTABLE_PRIMS = ("J1_6", "J1_7", "J1_8", "J2_6", "J2_7", "J2_8")
STAGE_ROOT = "/x_trainer_asm_0226_SLDASM"


def _mesh_path(stage: Usd.Stage, part_name: str) -> str:
    # The base layer keeps the assembly transforms under STAGE_ROOT while the
    # referenced visual meshes live in stage-level /visuals and /colliders
    # scopes.
    return f"/visuals/{part_name}/{part_name}/mesh"


def _matrix_rows(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangles(mesh: UsdGeom.Mesh) -> Iterable[tuple[int, int, int]]:
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or []]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or []]
    cursor = 0
    for count in counts:
        if count < 3:
            cursor += count
            continue
        face = indices[cursor : cursor + count]
        cursor += count
        # The official finger meshes are triangles. Fan triangulation also
        # keeps this extractor usable if a later USD revision contains n-gons.
        for index in range(1, len(face) - 1):
            yield face[0], face[index], face[index + 1]


def _normal(a: Gf.Vec3d, b: Gf.Vec3d, c: Gf.Vec3d) -> Gf.Vec3d:
    vector = Gf.Cross(b - a, c - a)
    length = math.sqrt(float(Gf.Dot(vector, vector)))
    if length <= 1e-15:
        return Gf.Vec3d(0.0, 0.0, 0.0)
    return vector / length


def _write_binary_stl(path: Path, name: str, points, triangles) -> None:
    """Write a binary STL because MuJoCo's STL loader expects that form."""
    header = f"Dobot X-Trainer USD mesh {name}".encode("ascii")[:80].ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for first, second, third in triangles:
            a = Gf.Vec3d(points[first])
            b = Gf.Vec3d(points[second])
            c = Gf.Vec3d(points[third])
            normal = _normal(a, b, c)
            stream.write(
                struct.pack(
                    "<12fH",
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(a[0]),
                    float(a[1]),
                    float(a[2]),
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    0,
                )
            )


def _write_obj(path: Path, name: str, points, triangles) -> None:
    """Write an OBJ with the USD prim-local coordinates unchanged."""
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"# Official Dobot X-Trainer USD mesh: {name}\n")
        stream.write("# Units: meters; frame: USD mesh prim-local\n")
        for point in points:
            stream.write("v %.9g %.9g %.9g\n" % tuple(float(value) for value in point))
        for first, second, third in triangles:
            stream.write(f"f {first + 1} {second + 1} {third + 1}\n")


def extract(stage_path: Path, output_dir: Path, part_names: tuple[str, ...]) -> dict:
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {stage_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    stage_root = stage.GetPrimAtPath(STAGE_ROOT)
    if not stage_root:
        raise RuntimeError(f"Missing USD stage root {STAGE_ROOT}")

    manifest = {
        "source": {
            "dataset": "Dobot-robot/X-trainer",
            "revision": "02f008252001f654e28dee40d615c713ff79935a",
            "stage": str(stage_path.relative_to(REPO_ROOT)),
            "stage_sha256": _hash_file(stage_path),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        },
        "frame": {
            "export_frame": "each_mesh_prim_local",
            "note": (
                "The USD assembly transform is recorded per part. It is not "
                "a Nova2 Link6 or flange transform."
            ),
        },
        "parts": [],
    }

    for part_name in part_names:
        mesh_prim = stage.GetPrimAtPath(_mesh_path(stage, part_name))
        if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Missing USD mesh prim for {part_name}")
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get() or []
        triangles = list(_triangles(mesh))
        if not points or not triangles:
            raise RuntimeError(f"Empty USD mesh for {part_name}")

        output_path = output_dir / f"{part_name}_local.stl"
        _write_binary_stl(output_path, part_name, points, triangles)
        obj_path = output_dir / f"{part_name}_local.obj"
        _write_obj(obj_path, part_name, points, triangles)
        local_extent = mesh.GetExtentAttr().Get()
        part_prim = stage.GetPrimAtPath(f"{STAGE_ROOT}/{part_name}")
        world_matrix = xform_cache.GetLocalToWorldTransform(part_prim)
        manifest["parts"].append(
            {
                "name": part_name,
                "prim": str(mesh_prim.GetPath()),
                "export": str(output_path.relative_to(REPO_ROOT)),
                "obj_export": str(obj_path.relative_to(REPO_ROOT)),
                "mesh_vertices": len(points),
                "mesh_triangles": len(triangles),
                "local_extent_m": [
                    [float(value) for value in local_extent[0]],
                    [float(value) for value in local_extent[1]],
                ],
                "assembly_prim_world_matrix": _matrix_rows(world_matrix),
                "assembly_prim_world_translation_m": [
                    float(value) for value in world_matrix.ExtractTranslation()
                ],
                "sha256": _hash_file(output_path),
                "obj_sha256": _hash_file(obj_path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--part",
        dest="parts",
        action="append",
        choices=EXTRACTABLE_PRIMS,
        help="USD part to extract; may be repeated (default: J1_7 and J1_8)",
    )
    args = parser.parse_args()
    part_names = tuple(args.parts) if args.parts else DEFAULT_PRIMS
    manifest = extract(args.stage.resolve(), args.output_dir.resolve(), part_names)
    print(f"Extracted {len(manifest['parts'])} official X-Trainer finger meshes")
    for part in manifest["parts"]:
        print(
            f"{part['name']}: {part['mesh_vertices']} vertices, "
            f"{part['mesh_triangles']} triangles, {part['sha256']}"
        )
    print(f"Manifest: {args.output_dir.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
