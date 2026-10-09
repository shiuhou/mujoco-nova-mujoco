"""S5B uses official mock Responses output offline; no scripted object path."""

import json

import numpy as np
import pytest

from s5_agent import make_replay_policy
from s5_benchmark import (
    BenchmarkRecorder,
    benchmark_action_limits,
    require_s5a_pass,
    task_failure,
)
from s5_common import (
    BENCHMARK_INSTRUCTION,
    BenchmarkRunLimits,
    Budget,
    RunLimits,
    S5Error,
)
from s5_embodiment import ActionLimits, NovaMujocoEmbodiment
from s5_loop import run_loop
from test_s5_closed_loop import chunk, policy_with_reply, response
from inspect_robots.scene import Scene


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_llm_calls", 13),
        ("max_decisions", 13),
        ("max_episode_steps", 385),
        ("max_wall_time", 601),
        ("max_wall_time", float("nan")),
    ],
)
def test_benchmark_ceiling_cannot_expand(field, value):
    with pytest.raises(ValueError):
        BenchmarkRunLimits(**{field: value})


def test_benchmark_keeps_s5a_budgets_separate():
    with pytest.raises(ValueError):
        RunLimits(max_llm_calls=12)
    assert BenchmarkRunLimits().max_llm_calls == 12
    mission = benchmark_action_limits()
    safe = ActionLimits()
    assert mission.workspace_low == safe.workspace_low
    assert mission.workspace_high == safe.workspace_high
    assert mission.waypoint_displacement_m == safe.waypoint_displacement_m == 0.02
    assert not mission.open_only and safe.open_only
    assert mission.max_waypoints == 32
    require_s5a_pass()


@pytest.fixture
def mission():
    env = NovaMujocoEmbodiment(
        Budget(BenchmarkRunLimits(max_llm_calls=2, max_decisions=2)),
        benchmark_action_limits(),
    )
    try:
        yield env
    finally:
        env.close()


def test_benchmark_close_command_uses_same_executor_and_fresh_observation(mission):
    obs = mission.reset(Scene(id="test", instruction=BENCHMARK_INSTRUCTION))
    mission.begin_decision(obs)
    action = np.r_[obs.state["eef_state"][:3], 0, 0, 0, 0]
    fresh = mission.execute_chunk(chunk([action]), obs).observation
    assert fresh.state_time > obs.state_time
    assert mission.action_trace[-1]["executor_status"] == "completed"
    assert mission.sim.get_gripper_state()["commanded_normalized_gripper"] == 0
    assert set(fresh.state) == {"eef_state"}
    assert set(fresh.extra) == {"env_step"}
    assert "0.150" in mission.info.docs
    assert "S5A requires" not in mission.info.docs


@pytest.mark.parametrize(
    "index,value", [(3, 0.1), (6, 1.01), (6, -0.1), (0, float("nan")), (0, 1)]
)
def test_benchmark_preserves_validation_before_mutation(mission, index, value):
    obs = mission.reset(Scene(id="test", instruction=BENCHMARK_INSTRUCTION))
    mission.begin_decision(obs)
    q = mission.sim.data.qpos.copy()
    action = obs.state["eef_state"].copy()
    action[index] = value
    with pytest.raises(S5Error):
        mission.execute_chunk(chunk([action]), obs)
    np.testing.assert_array_equal(q, mission.sim.data.qpos)
    assert mission.budget.steps == 0


def test_official_benchmark_loop_budget_and_model_input(mission, tmp_path):
    captures = []

    def reply(request):
        captures.append(json.loads(request.content))
        return response(
            "move_to",
            {
                "targets": {
                    "z": float(mission.current.state["eef_state"][2]) + 0.002,
                    "gripper": 1,
                },
                "note": "offline bounded fixture",
            },
        )

    policy, transport = policy_with_reply(mission, reply)
    recorder = BenchmarkRecorder(tmp_path)
    try:
        result = run_loop(mission, policy, transport, benchmark=recorder)
    finally:
        policy._client.close()
    assert result["failure_category"] == "max_calls_reached"
    assert result["benchmark_success"] is False
    assert result["decisions"] == result["provider_attempts"] == 2
    assert [row["generation"] for row in mission.observation_trace] == [1, 2, 3]
    assert all(row["fresh_after_settle"] for row in result["events"])
    assert (tmp_path / "initial_montage.png").exists() and (
        tmp_path / "final_montage.png"
    ).exists()
    wire = json.dumps(captures)
    assert BENCHMARK_INSTRUCTION in wire
    for privileged in (
        "final_blue_pose",
        "inside_target",
        "object_poses_evaluator_only",
        "bilateral_blue_contact",
        "holding_now",
    ):
        assert privileged not in wire
    assert len(recorder.records) == 4
    assert result["evaluation_only"]["task_failure_category"] == "approach_fail"


def test_model_done_cannot_claim_benchmark_success(mission):
    policy, transport = policy_with_reply(
        mission, lambda r: response("done", {"note": "I succeeded"})
    )
    try:
        result = run_loop(mission, policy, transport, benchmark=BenchmarkRecorder())
    finally:
        policy._client.close()
    assert result["benchmark_success"] is False
    assert result["action_count"] == 0
    assert result["failure_category"] == "approach_fail"
    assert not result["evaluation_only"]["inside_target"]


def test_decision_budget_can_stop_before_call_budget(mission):
    mission.budget = Budget(BenchmarkRunLimits(max_llm_calls=2, max_decisions=1))
    policy, transport = make_replay_policy(mission, mission.budget)
    try:
        result = run_loop(mission, policy, transport, benchmark=BenchmarkRecorder())
    finally:
        policy._client.close()
    assert result["decisions"] == result["provider_attempts"] == 1
    assert result["failure_category"] == "max_steps_reached"


@pytest.mark.parametrize(
    "updates,closed,category",
    [
        ({}, False, "approach_fail"),
        ({}, True, "grasp_fail"),
        ({"object_grasped": True, "holding_now": True}, True, "transport_fail"),
        ({"object_grasped": True}, True, "place_miss"),
        (
            {"object_grasped": True, "inside_target": True, "holding_now": True},
            True,
            "release_fail",
        ),
        ({"robot_faulted": True}, False, "validation_abort"),
    ],
)
def test_task_failure_uses_evaluator_facts(updates, closed, category):
    metrics = {
        "robot_faulted": False,
        "object_grasped": False,
        "inside_target": False,
        "holding_now": False,
        "released": False,
    }
    assert task_failure({**metrics, **updates}, closed_command=closed) == category
