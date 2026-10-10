"""S4R numerical repeatability must not mask wrong or stale observations."""

from copy import deepcopy

import numpy as np
import pytest

from nova_cameras import CAMERA_NAMES, NovaMujocoCameraRig, SimObservation
from run_s4_camera_validation import run
from run_s4_render_stress import pixel_difference
from s4_validation import observation_repeat_report, rgb_repeat_report


def observation(generation):
    frames = {
        name: {
            "state_hash": "identical-state",
            "frame_generation": generation,
            "state_time": 0.0,
            "sim_time": 0.0,
            "camera_world_position_m": [float(i), 0.0, 0.0],
            "camera_to_world_rotation": np.eye(3).tolist(),
        }
        for i, name in enumerate(CAMERA_NAMES)
    }
    return SimObservation(
        images={
            name: np.full((32, 32, 3), 50 + i * 20, np.uint8)
            for i, name in enumerate(CAMERA_NAMES)
        },
        state={"eef_state": np.zeros(7)},
        image_times=dict.fromkeys(CAMERA_NAMES, 0.0),
        state_time=0.0,
        extra={
            "nova_mujoco": {
                "state_hash": "identical-state",
                "frame_generation": generation,
                "camera": {"frames": frames},
            }
        },
    )


def test_observed_four_pixel_one_lsb_residual_is_reported_not_erased():
    first, second = observation(1), observation(2)
    rgb = second.images["front_rgb"]
    rgb[20:22, 20:22, 0] += 1
    rgb[20, 20:22, 1] += 1
    report = observation_repeat_report(first, second)
    assert report["passed"]
    pixel = report["pixels"]["front_rgb"]
    assert not pixel["exact"]
    assert pixel["differing_pixels"] == 4
    assert pixel["max_channel_difference"] == 1
    assert pixel["mean_absolute_difference"] == 6 / (32 * 32 * 3)
    assert pixel["bbox_xyxy_inclusive"] == [20, 20, 21, 21]
    assert pixel_difference(first.images["front_rgb"], rgb)["differing_channels"] == 6
    assert rgb[20, 20, 0] == 51  # Validation never modifies RGB.


@pytest.mark.parametrize(
    "defect",
    [
        "intensity",
        "extent",
        "swapped",
        "stale",
        "state_hash",
        "state_values",
        "camera",
        "frame_generation",
        "frame_time",
        "image_time",
        "shape",
        "dtype",
        "missing_camera",
    ],
)
def test_reject_wrong_stale_or_excessively_different_observation(defect):
    first, second = observation(1), observation(2)
    meta = second.extra["nova_mujoco"]
    if defect == "intensity":
        second.images["front_rgb"][0, 0, 0] += 2
    elif defect == "extent":
        second.images["front_rgb"][0, :9, 0] += 1
    elif defect == "swapped":
        second.images["front_rgb"], second.images["right_rgb"] = (
            second.images["right_rgb"],
            second.images["front_rgb"],
        )
    elif defect == "stale":
        second = deepcopy(first)
    elif defect == "state_hash":
        meta["state_hash"] = "changed-state"
        for frame in meta["camera"]["frames"].values():
            frame["state_hash"] = "changed-state"
    elif defect == "state_values":
        second.state["eef_state"][0] = 0.001
    elif defect == "camera":
        meta["camera"]["frames"]["front_rgb"]["camera_world_position_m"][0] += 0.001
    elif defect == "frame_generation":
        meta["camera"]["frames"]["front_rgb"]["frame_generation"] = 1
    elif defect == "frame_time":
        meta["camera"]["frames"]["front_rgb"]["sim_time"] = 0.1
    elif defect == "image_time":
        second.image_times["front_rgb"] = 0.1
    elif defect == "shape":
        second.images["front_rgb"] = np.zeros((33, 32, 3), np.uint8)
    elif defect == "dtype":
        second.images["front_rgb"] = second.images["front_rgb"].astype(float)
    elif defect == "missing_camera":
        del second.images["front_rgb"]
    assert not observation_repeat_report(first, second)["passed"]


def test_exact_rgb_remains_exact():
    rgb = np.zeros((32, 32, 3), np.uint8)
    assert rgb_repeat_report(rgb, rgb)["exact"]
    assert rgb_repeat_report(rgb, rgb)["passed"]


def test_s4_movement_gate_rejects_cached_pixels_with_new_metadata(
    tmp_path, monkeypatch
):
    draw = NovaMujocoCameraRig._draw
    cached = {}

    def stale_draw(self, name):
        if name not in cached:
            cached[name] = draw(self, name)
        return cached[name]

    monkeypatch.setattr(NovaMujocoCameraRig, "_draw", stale_draw)
    report = run(tmp_path / "stale", repetitions=1)
    assert report["status"] == "FAIL"
    assert report["static_repeat_validation"]["passed"]
    assert report["reset_repeat_validation"]["passed"]
    assert all(
        v == 0
        for row in report["movement_checks"]
        for v in row["changed_pixel_fraction"].values()
    )
