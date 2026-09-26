#!/usr/bin/env python3
"""Audit Nova2 collision proxies against the official STL mesh bounds.

This tool is geometry-only. It does not run a physics simulation and does not
change ``nova2.xml``. The Phase 1 collision geoms are intentionally simple
boxes; this report verifies that every box encloses the corresponding official
visual mesh in the link's local frame and records how conservative the proxy
is.
"""

from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "nova2.xml"
ASSET_DIR = ROOT / "assets"
DEFAULT_TOLERANCE_M = 2e-6


@dataclass(frozen=True)
class CollisionReview:
    link: str
    mesh_file: str
    triangle_count: int
    mesh_min_m: tuple[float, float, float]
    mesh_max_m: tuple[float, float, float]
    proxy_center_m: tuple[float, float, float]
    proxy_half_size_m: tuple[float, float, float]
    boundary_error_min_m: tuple[float, float, float]
    boundary_error_max_m: tuple[float, float, float]
    mesh_aabb_volume_m3: float
    proxy_box_volume_m3: float
    proxy_volume_ratio: float
    encloses_mesh: bool


def _vector(text: str | None, *, label: str) -> np.ndarray:
    if text is None:
        return np.zeros(3)
    values = np.asarray([float(value) for value in text.split()], dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain three finite values")
    return values


def _binary_stl_bounds(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if expected_size != len(data):
        raise ValueError(
            f"expected binary STL size {expected_size} for {path}, got {len(data)}"
        )
    vertices = []
    for offset in (12, 24, 36):
        vertices.append(
            np.ndarray(
                shape=(triangle_count, 3),
                dtype="<f4",
                buffer=data,
                offset=84 + offset,
                strides=(50, 4),
            )
        )
    all_vertices = np.concatenate(vertices, axis=0)
    if not np.all(np.isfinite(all_vertices)):
        raise ValueError(f"STL contains non-finite vertices: {path}")
    return triangle_count, all_vertices.min(axis=0), all_vertices.max(axis=0)


def _iter_bodies(element: ET.Element) -> Iterable[ET.Element]:
    for body in element.findall("body"):
        yield body
        yield from _iter_bodies(body)


def _tuple(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


def review_model(
    model_path: Path = MODEL_PATH,
    asset_dir: Path = ASSET_DIR,
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> list[CollisionReview]:
    """Return one proxy audit row for each visual mesh in the MJCF."""

    if tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError("tolerance_m must be finite and non-negative")
    root = ET.parse(model_path).getroot()
    mesh_files = {
        mesh.get("name"): mesh.get("file") for mesh in root.findall("./asset/mesh")
    }
    rows: list[CollisionReview] = []
    for body in _iter_bodies(root.find("worldbody")):
        visual_meshes = [
            geom
            for geom in body.findall("geom")
            if geom.get("type") == "mesh" and geom.get("mesh")
        ]
        if not visual_meshes:
            continue
        if len(visual_meshes) != 1:
            raise ValueError(
                f"expected one visual mesh in body {body.get('name')!r}, "
                f"found {len(visual_meshes)}"
            )
        visual = visual_meshes[0]
        link = body.get("name") or "<unnamed>"
        mesh_file = mesh_files.get(visual.get("mesh"))
        if mesh_file is None:
            raise ValueError(f"visual mesh asset is missing for body {link!r}")
        collision = next(
            (
                geom
                for geom in body.findall("geom")
                if geom.get("type") == "box"
                and geom.get("name", "").endswith("_collision")
            ),
            None,
        )
        if collision is None:
            raise ValueError(f"collision box is missing for body {link!r}")

        path = asset_dir / mesh_file
        triangle_count, mesh_min, mesh_max = _binary_stl_bounds(path)
        proxy_center = _vector(collision.get("pos"), label=f"{link} collision pos")
        proxy_half = _vector(collision.get("size"), label=f"{link} collision size")
        if np.any(proxy_half <= 0):
            raise ValueError(f"collision half-size must be positive for {link!r}")
        proxy_min = proxy_center - proxy_half
        proxy_max = proxy_center + proxy_half
        boundary_error_min = proxy_min - mesh_min
        boundary_error_max = proxy_max - mesh_max
        encloses = bool(
            np.all(proxy_min <= mesh_min + tolerance_m)
            and np.all(proxy_max >= mesh_max - tolerance_m)
        )
        mesh_dimensions = mesh_max - mesh_min
        proxy_dimensions = 2.0 * proxy_half
        mesh_volume = float(np.prod(mesh_dimensions))
        proxy_volume = float(np.prod(proxy_dimensions))
        rows.append(
            CollisionReview(
                link=link,
                mesh_file=mesh_file,
                triangle_count=triangle_count,
                mesh_min_m=_tuple(mesh_min),
                mesh_max_m=_tuple(mesh_max),
                proxy_center_m=_tuple(proxy_center),
                proxy_half_size_m=_tuple(proxy_half),
                boundary_error_min_m=_tuple(boundary_error_min),
                boundary_error_max_m=_tuple(boundary_error_max),
                mesh_aabb_volume_m3=mesh_volume,
                proxy_box_volume_m3=proxy_volume,
                proxy_volume_ratio=proxy_volume / mesh_volume,
                encloses_mesh=encloses,
            )
        )
    return rows


def _print_report(rows: list[CollisionReview], tolerance_m: float) -> None:
    print(f"model: {MODEL_PATH.name}")
    print(f"tolerance_m: {tolerance_m:g}")
    print("link mesh triangles encloses max_boundary_error_mm proxy_aabb_ratio")
    for row in rows:
        errors = np.abs(np.asarray(row.boundary_error_min_m + row.boundary_error_max_m))
        print(
            f"{row.link} {row.mesh_file} {row.triangle_count} "
            f"{str(row.encloses_mesh).lower()} {np.max(errors) * 1000:.6f} "
            f"{row.proxy_volume_ratio:.6f}"
        )
    print(f"visual_meshes: {len(rows)}")
    print(f"enclosing_proxies: {sum(row.encloses_mesh for row in rows)}")
    if rows and all(row.encloses_mesh for row in rows):
        print("status: all collision boxes enclose their official visual mesh AABBs")
        print("note: AABB proxies are conservative and may produce false contacts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--tolerance-m", type=float, default=DEFAULT_TOLERANCE_M)
    parser.add_argument("--json", action="store_true", help="print JSON rows")
    args = parser.parse_args()
    rows = review_model(args.model, args.asset_dir, tolerance_m=args.tolerance_m)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        _print_report(rows, args.tolerance_m)
    return 0 if rows and all(row.encloses_mesh for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
