"""Ground-truth projections and simple color diagnostics for S4 test fixtures."""

from __future__ import annotations

import numpy as np
import mujoco

from nova_cameras import CAMERA_NAMES, NovaMujocoCameraRig, SimObservation

FIXTURE_NAMES = ("s4_blue_box", "s4_black_box", "s4_red_target")


def rgb_repeat_report(reference: np.ndarray, image: np.ndarray) -> dict:
    """Same-state visual repeatability, not an observation freshness predicate.

    S4R EGL/MSAA evidence found 1-LSB changes in four pixels with frozen inputs
    and up to five after motion/reset. Keep a narrow eight-pixel ceiling,
    independent of resolution. Never blur, resize, quantize or alter policy RGB.
    Larger differences fail closed; this is not a cross-driver equivalence rule.
    """
    if (
        reference.shape != image.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
        or reference.size == 0
        or reference.dtype != np.uint8
        or image.dtype != np.uint8
    ):
        return {"passed": False, "exact": False, "reason": "invalid_rgb_contract"}
    delta = np.abs(reference.astype(np.int16) - image.astype(np.int16))
    changed = np.any(delta != 0, axis=2)
    y, x = np.nonzero(changed)
    count = int(changed.sum())
    maximum = int(delta.max())
    return {
        "passed": count <= 8 and maximum <= 1,
        "exact": count == 0,
        "differing_pixels": count,
        "max_channel_difference": maximum,
        "mean_absolute_difference": float(delta.mean()),
        "bbox_xyxy_inclusive": (
            [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if count else None
        ),
        "limits": {"differing_pixels": 8, "max_channel_difference": 1},
    }


def observation_repeat_report(first: SimObservation, second: SimObservation) -> dict:
    """Require a new generation of the exact same state and camera transforms.

    Only for same-rig static/reset comparisons. Movement must separately change
    the state hash and images, as checked by the S4 movement and S5 freshness gates.
    """
    a, b = (obs.extra["nova_mujoco"] for obs in (first, second))
    frames_a, frames_b = (meta["camera"]["frames"] for meta in (a, b))
    contract = (
        set(first.images) == set(second.images) == set(CAMERA_NAMES)
        and set(frames_a) == set(frames_b) == set(CAMERA_NAMES)
        and set(first.state) == set(second.state)
    )
    if not contract:
        return {"passed": False, "reason": "observation_contract_mismatch"}
    checks = {
        "state_hash_equal": a["state_hash"] == b["state_hash"],
        "state_values_equal": all(
            np.array_equal(first.state[key], second.state[key]) for key in first.state
        ),
        "time_equal": first.state_time == second.state_time,
        "generation_advanced": b["frame_generation"] > a["frame_generation"],
        "camera_poses_equal": all(
            frames_a[name][key] == frames_b[name][key]
            for name in CAMERA_NAMES
            for key in ("camera_world_position_m", "camera_to_world_rotation")
        ),
        "frame_metadata_coherent": all(
            frame["state_hash"] == meta["state_hash"]
            and frame["frame_generation"] == meta["frame_generation"]
            and frame["state_time"] == frame["sim_time"] == obs.state_time
            and obs.image_times.get(name) == obs.state_time
            for obs, meta, frames in ((first, a, frames_a), (second, b, frames_b))
            for name, frame in frames.items()
        ),
    }
    pixels = {
        name: rgb_repeat_report(first.images[name], second.images[name])
        for name in CAMERA_NAMES
    }
    return {
        "passed": all(checks.values()) and all(p["passed"] for p in pixels.values()),
        "checks": checks,
        "pixels": pixels,
    }


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
