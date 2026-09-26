from __future__ import annotations

import yaml

from analyze_gripper_calibration import analyze


def test_template_is_an_explicit_gate() -> None:
    with open("gripper_calibration.yaml", encoding="utf-8") as stream:
        record = yaml.safe_load(stream)
    assert record["flange_to_gripper"]["physically_verified"] is False
    assert record["flange_to_gripper"]["translation_mm"] is None
    assert "Nova wrist" in record["flange_to_gripper"]["usd_candidate"]["reason"]


def test_calibration_analysis_fits_measured_fixture() -> None:
    record = {
        "samples": [
            {
                "command": 1470,
                "left_position_mm": 0,
                "right_position_mm": 0,
                "gap_mm": 0,
            },
            {
                "command": 1980,
                "left_position_mm": 5,
                "right_position_mm": -5,
                "gap_mm": 10,
            },
            {
                "command": 2490,
                "left_position_mm": 10,
                "right_position_mm": -10,
                "gap_mm": 20,
            },
        ],
        "finger_mapping": {},
    }
    report = analyze(record)
    assert "left_travel_mm: 10.000000" in report
    assert "right_travel_mm: -10.000000" in report
    assert "approximate_symmetry: yes" in report
    assert "left_fit_max_abs_residual_mm: 0.000000" in report
