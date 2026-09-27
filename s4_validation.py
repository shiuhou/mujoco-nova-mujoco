"""Ground-truth projections and simple color diagnostics for S4 test fixtures."""

from __future__ import annotations

import numpy as np
import mujoco

from nova_cameras import CAMERA_NAMES, NovaMujocoCameraRig, SimObservation

FIXTURE_NAMES = ("s4_blue_box", "s4_black_box", "s4_red_target")


def fixture_report(rig: NovaMujocoCameraRig, observation: SimObservation) -> dict:
    """Check known surface points, not a learned detector or real calibration."""
    points = []
    for name in FIXTURE_NAMES:
        geom_id = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing S4 fixture {name}")
        points.append(
            rig.data.geom_xpos[geom_id] + [0, 0, rig.model.geom_size[geom_id, 2]]
        )
    report = {}
    for name in CAMERA_NAMES:
        pixels, depths = rig.project(name, points)
        rgb = observation.images[name]
        entries = {}
        for fixture, uv, depth in zip(FIXTURE_NAMES, pixels, depths, strict=True):
            inside = bool(
                rig.renderer_config.near_m < depth < rig.renderer_config.far_m
                and 5 <= uv[0] < rgb.shape[1] - 5
                and 5 <= uv[1] < rgb.shape[0] - 5
            )
            color = np.zeros(3)
            if inside:
                u, v = np.rint(uv).astype(int)
                color = np.median(rgb[v - 2 : v + 3, u - 2 : u + 3], axis=(0, 1))
            r, g, b = (float(value) for value in color)
            if fixture == "s4_blue_box":
                distinct = b > 50 and b > r + 30 and b > g + 30
            elif fixture == "s4_red_target":
                distinct = r > 50 and r > b + 30 and r > g + 30
            else:
                distinct = max(r, g, b) < 65
            entries[fixture] = {
                "pixel_uv": uv.tolist(),
                "view_depth_m": float(depth),
                "inside_frustum": inside,
                "patch_median_rgb": color.tolist(),
                "color_visible": bool(inside and distinct),
            }
        report[name] = entries
    return report
