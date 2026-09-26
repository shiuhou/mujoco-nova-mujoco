from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from capture_real_pose import (
    ALLOWED_COMMAND_SET,
    DashboardClient,
    DashboardProtocolError,
    Snapshot,
    _load_dataset,
    _sample_document,
    _write_dataset,
    parse_dashboard_response,
    read_snapshot,
    snapshot_rejection_reason,
)
from compare_pose import (
    AXIS_HYPOTHESES,
    PoseComparison,
    _orientation_error_deg,
    _record_values,
    calculate_statistics,
    compare_samples,
    diagnostic_messages,
    load_samples,
)


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "synthetic_multi_samples.json"


class FakeSocket:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.timeout: float | None = None
        self.closed = False

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, _size: int) -> bytes:
        if not self.responses:
            return b""
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _fixture_document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _dashboard_fixture_socket() -> FakeSocket:
    return FakeSocket(
        [
            b"0,{5},RobotMode();",
            b"0,{[]},GetErrorID();",
            b"0,{1,2,3,4,5,6},GetAngle();",
            b"0,{100,200,300,0,0,90},GetPose(user=0,tool=0);",
        ]
    )


def test_capture_parser_and_exact_read_only_command_sequence() -> None:
    socket = _dashboard_fixture_socket()
    client = DashboardClient(socket)
    snapshot = read_snapshot(client)

    assert socket.sent == [
        command.encode()
        for command in (
            "RobotMode()",
            "GetErrorID()",
            "GetAngle()",
            "GetPose(user=0,tool=0)",
        )
    ]
    assert set(socket.sent) <= {command.encode() for command in ALLOWED_COMMAND_SET}
    assert snapshot.robot_mode == 5
    assert snapshot.error_ids == ()
    assert snapshot.q_deg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert snapshot.pose_real_mm_deg == (100.0, 200.0, 300.0, 0.0, 0.0, 90.0)


def test_capture_rejects_non_allowlisted_command_and_malformed_response() -> None:
    client = DashboardClient(_dashboard_fixture_socket())
    with pytest.raises(PermissionError):
        client.query("GetPose()")
    with pytest.raises(DashboardProtocolError):
        parse_dashboard_response("0,{5},RobotMode()")


def test_capture_refuses_alarm_or_motion_states() -> None:
    fixture = _dashboard_fixture_socket()
    fixture.responses[0] = b"0,{9},RobotMode();"
    client = DashboardClient(fixture)
    snapshot = read_snapshot(client)
    assert "not a stable sample" in snapshot_rejection_reason(snapshot)


def test_capture_append_preserves_existing_samples(tmp_path: Path) -> None:
    output = tmp_path / "samples.json"
    existing = _load_dataset(output, overwrite=False)
    existing["samples"].append(
        {
            "name": "pose_01",
            "q_deg": [0, 0, 0, 0, 0, 0],
            "pose_real_mm_deg": [0, 0, 0, 0, 0, 0],
            "orientation": "rotation_vector",
            "user": 0,
            "tool": 0,
        }
    )
    _write_dataset(output, existing)
    loaded = _load_dataset(output, overwrite=False)
    loaded["samples"].append(
        _sample_document(
            "pose_02",
            Snapshot(
                robot_mode=5,
                error_ids=(),
                q_deg=(1, 2, 3, 4, 5, 6),
                pose_real_mm_deg=(10, 20, 30, 0, 0, 0),
                raw_responses={},
            ),
        )
    )
    _write_dataset(output, loaded)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [sample["name"] for sample in document["samples"]] == ["pose_01", "pose_02"]


def test_multi_sample_loader_and_current_legacy_comparison() -> None:
    samples = load_samples(_fixture_document())
    assert len(samples) == 3
    comparisons = compare_samples(samples)
    assert len(comparisons) == 6
    assert {item.axis_mode for item in comparisons} == set(AXIS_HYPOTHESES)
    assert {item.sample for item in comparisons} == {
        "synthetic_pose_01",
        "synthetic_pose_02",
        "synthetic_pose_03",
    }


def test_orientation_override_reinterprets_legacy_capture_metadata() -> None:
    document = _fixture_document()
    document["metadata"]["orientation"] = "rotation_vector"
    document["metadata"]["pose_orientation_convention"] = "rotation_vector"
    samples = load_samples(document, orientation_override="rpy")
    assert all(sample.orientation_convention == "rpy" for sample in samples)


def test_sample_orientation_takes_precedence_over_older_dataset_metadata() -> None:
    document = _fixture_document()
    document["metadata"]["orientation"] = "rotation_vector"
    document["samples"][1]["orientation"] = "rpy"
    samples = load_samples(document)
    assert [sample.orientation_convention for sample in samples] == [
        "rotation_vector",
        "rpy",
        "rotation_vector",
    ]


def test_legacy_single_record_format_remains_supported() -> None:
    q_deg, position_mm, rotation, convention = _record_values(
        {
            "user": 0,
            "tool": 0,
            "q_deg": [1, 2, 3, 4, 5, 6],
            "pose_mm_deg": [10, 20, 30, 0, 0, 90],
            "orientation": "rotation_vector",
        }
    )
    np.testing.assert_allclose(q_deg, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(position_mm, [10, 20, 30])
    assert convention == "rotation_vector"
    assert _orientation_error_deg(np.eye(3), rotation) == pytest.approx(90.0)


def test_loader_reports_malformed_and_missing_samples() -> None:
    malformed = _fixture_document()
    malformed["samples"][1]["q_deg"] = [1, 2]
    with pytest.raises(ValueError, match="exactly 6"):
        load_samples(malformed)

    missing = _fixture_document()
    del missing["samples"][0]["pose_real_mm_deg"]
    with pytest.raises(ValueError, match="pose_real_mm_deg"):
        load_samples(missing)


def test_relative_orientation_angle_and_statistics() -> None:
    identity = np.eye(3)
    ninety_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _orientation_error_deg(identity, ninety_z) == pytest.approx(90.0)
    assert _orientation_error_deg(ninety_z, identity) == pytest.approx(90.0)

    comparisons = [
        PoseComparison(
            "a", "current", np.zeros(3), np.zeros(3), np.array([3.0, 4, 0]), 5.0, 10.0
        ),
        PoseComparison(
            "b", "current", np.zeros(3), np.zeros(3), np.array([0.0, 0, 0]), 0.0, 20.0
        ),
    ]
    stats = calculate_statistics(comparisons)
    assert stats["current"]["position_error_mm"]["mean"] == pytest.approx(2.5)
    assert stats["current"]["position_error_mm"]["median"] == pytest.approx(2.5)
    assert stats["current"]["position_error_mm"]["rms"] == pytest.approx(np.sqrt(12.5))
    assert stats["current"]["position_error_mm"]["max"] == pytest.approx(5.0)
    assert stats["current"]["orientation_error_deg"]["rms"] == pytest.approx(
        np.sqrt(250.0)
    )


def test_diagnostics_are_non_mutating_and_report_offset_pattern() -> None:
    samples = load_samples(_fixture_document())
    comparisons = [
        PoseComparison(
            sample.name,
            "current",
            sample.real_position_mm,
            sample.real_position_mm + np.array([10.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            10.0,
            1.0,
        )
        for sample in samples
    ]
    messages = diagnostic_messages(samples, comparisons)
    assert any("constant TCP/base/tool-frame offset" in message for message in messages)
