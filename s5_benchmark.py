"""Evaluator-only S5B observations and artifacts; never passed to the policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from s5_common import ROOT, S5Error
from s5_embodiment import ActionLimits

S5A_HEAD = "5fb7d38753ffb6158abf29e331b87aeb31f30e67"
S5A_EVIDENCE = ROOT / "evidence/sim_s5a/20261009T124559Z-live"


def require_s5a_pass() -> None:
    validation = json.loads((S5A_EVIDENCE / "validation.json").read_text())
    result = json.loads((S5A_EVIDENCE / "live_agent_result.json").read_text())
    if not (
        validation.get("live_s5a_pass")
        and result.get("status") == "PASS"
        and all(result.get("checks", {}).values())
        and result.get("transmitted_requests", 0) > 0
    ):
        raise S5Error("precondition_failure", "live S5A PASS required before S5B")


def benchmark_action_limits() -> ActionLimits:
    # High-level mission bounds, chosen independently of hidden target poses.
    # Frozen workspace, 20 mm waypoint bound and every IK check remain active.
    return ActionLimits(
        chunk_path_m=0.15, episode_path_m=1.8, max_waypoints=32, open_only=False
    )


def task_failure(metrics: dict, *, closed_command: bool) -> str:
    if metrics["robot_faulted"]:
        return "validation_abort"
    if not metrics["object_grasped"]:
        return "grasp_fail" if closed_command else "approach_fail"
    if not metrics["inside_target"]:
        return "transport_fail" if metrics["holding_now"] else "place_miss"
    if metrics["holding_now"] or not metrics["released"]:
        return "release_fail"
    return "place_miss"  # Includes an unsettled placement or disturbed distractor.


class BenchmarkRecorder:
    """Read-only evaluator and RGB artifact sink, with no action generation."""

    def __init__(self, output: Path | None = None):
        self.output = output
        self.records: list[dict] = []
        self.closed_command = False
        self.transported = False
        self.dropped = False
        self._grasp_xy: np.ndarray | None = None
        self._previous_holding = False
        self._saved: set[str] = set()

    def montage(self, observation, name: str) -> None:
        if self.output is None or observation is None:
            return
        canvas = Image.new("RGB", (960, 290), "white")
        draw = ImageDraw.Draw(canvas)
        for i, camera in enumerate(("front_rgb", "right_rgb", "wrist_rgb")):
            frame = Image.fromarray(observation.images[camera])
            frame.thumbnail((320, 240))
            canvas.paste(frame, (i * 320, 30))
            draw.text((i * 320 + 8, 8), camera, fill="black")
        draw.text(
            (8, 272),
            f"S5B {name}; simulation time {observation.state_time:.3f}s",
            fill="black",
        )
        canvas.save(self.output / f"{name}_montage.png")

    def capture(self, embodiment, label: str) -> dict:
        metrics = embodiment.sim.compute_manipulation_metrics()
        gs = embodiment.sim.get_gripper_state()
        self.closed_command |= gs["commanded_normalized_gripper"] < 0.5
        xy = embodiment.sim.get_object_pose("blue_block").position_m[:2]
        if metrics["holding_now"]:
            if self._grasp_xy is None:
                self._grasp_xy = xy.copy()
            self.transported |= bool(np.linalg.norm(xy - self._grasp_xy) > 0.02)
        if (
            self._previous_holding
            and not metrics["holding_now"]
            and not metrics["inside_target"]
            and gs["commanded_normalized_gripper"] < 0.5
        ):
            self.dropped = True
        self._previous_holding = metrics["holding_now"]
        row = {
            "label": label,
            "generation": embodiment.observation_trace[-1]["generation"],
            "observation_available": embodiment.current is not None,
            "evaluation_only": metrics,
            "bilateral_blue_contact": gs["bilateral_blue_contact"],
            "gripper_command": gs["commanded_normalized_gripper"],
        }
        self.records.append(row)
        self.montage(embodiment.current, label)
        for stage, reached in (
            ("grasp", gs["bilateral_blue_contact"]),
            ("lifted", metrics["holding_now"]),
            ("transport", self.transported),
        ):
            if reached and stage not in self._saved:
                self.montage(embodiment.current, stage)
                self._saved.add(stage)
        return metrics

    def summary(self, metrics: dict, failure: str | None) -> dict:
        return {
            **metrics,
            "success": bool(
                metrics["success"] and not metrics["holding_now"] and failure is None
            ),
            "transport_observed": self.transported,
            "dropped_observed": self.dropped,
            "close_command_issued": self.closed_command,
            "grasp_contact_observed": any(
                row["bilateral_blue_contact"] for row in self.records
            ),
            "task_failure_category": None
            if metrics["success"]
            else "dropped"
            if self.dropped
            else task_failure(metrics, closed_command=self.closed_command),
            "target_selection": "blue_contact_observed"
            if any(row["bilateral_blue_contact"] for row in self.records)
            else "unverified",
            "evaluation_source": "unchanged S3 simulator, never supplied to policy",
        }
