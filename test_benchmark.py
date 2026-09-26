"""Offline benchmark artifact tests."""

from __future__ import annotations

import json

import pytest

import run_s1_s2_benchmark as benchmark


def test_benchmark_writes_structured_unique_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "VALIDATED_START_NAMES", ("pose_01",))
    monkeypatch.setattr(
        benchmark,
        "DIRECTIONS",
        {"+Z": benchmark.np.array([0.0, 0.0, 0.010])},
    )
    monkeypatch.setattr(
        benchmark,
        "_simulation_throughput",
        lambda: {
            "steps": 1,
            "simulation_time_s": 0.002,
            "host_elapsed_s": 0.001,
            "steps_per_host_second": 1000.0,
        },
    )
    output = tmp_path / "attempt-001"

    document = benchmark.run(output)

    assert document["summary"]["pass_count"] == 1
    assert document["cases"][0]["direction"] == "+Z"
    assert document["cases"][0]["action_frame"] == "base"
    assert document["cases"][0]["controlled_frame"] == "link6"
    assert document["side_effects"]["robot_connections"] == 0
    assert document["side_effects"]["robot_commands"] == 0
    assert json.loads((output / "benchmark.json").read_text())["schema"].endswith("_v1")
    assert (
        "simulator action-interface characterization"
        in (output / "summary.md").read_text()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        benchmark.run(output)
