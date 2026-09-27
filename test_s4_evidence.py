"""Structured S4 evidence must describe actual renders and refuse overwrites."""

import json
import pytest
from PIL import Image

from run_s4_camera_validation import run


def test_evidence_records_frames_and_side_effects_without_overwrite(tmp_path):
    output = tmp_path / "s4-evidence"
    doc = run(output, repetitions=1, debug=True)
    assert doc["status"] == "PASS"
    assert all(value == 0 for value in doc["side_effects"].values())
    assert all(value == 0 for value in doc["blocked_io_attempts"].values())
    assert set(doc["performance"]) == {
        "front_rgb",
        "right_rgb",
        "wrist_rgb",
        "all_three",
    }
    for name, shape in doc["shapes"].items():
        im = Image.open(output / f"{name}.png")
        assert im.mode == "RGB" and im.size == (shape[1], shape[0])
    assert json.loads((output / "validation.json").read_text())["status"] == "PASS"
    assert (output / "camera_debug.png").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(output, repetitions=1)
