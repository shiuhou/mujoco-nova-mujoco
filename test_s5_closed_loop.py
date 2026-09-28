"""Offline S5 boundary tests, including the actual official agent/Responses path."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import httpx
import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk
from inspect_robots_agent.policy import agent_policy
from PIL import Image

from s5_agent import (
    BoundedTransport,
    SimulationIOGuard,
    make_live_policy,
    make_replay_policy,
)
from s5_common import (
    ROOT,
    S5A_INSTRUCTION,
    Budget,
    RunLimits,
    S5Error,
    live_settings,
    verify_official_stack,
    wall_deadline,
)
from s5_embodiment import SHAPES, ActionLimits, NovaMujocoEmbodiment
from s5_loop import run_loop


@pytest.fixture
def embodiment():
    env = NovaMujocoEmbodiment(Budget(RunLimits()))
    try:
        env.reset(Scene(id="test", instruction=S5A_INSTRUCTION))
        yield env
    finally:
        env.close()


def vector(env, *, dz=0.002):
    return np.r_[env.sim.get_tool_pose().position_m + [0, 0, dz], 0, 0, 0, 1.0]


def chunk(vectors):
    return ActionChunk(actions=[Action(data=np.asarray(v)) for v in vectors])


def test_observation_exact_contract_and_no_privileged_state(embodiment):
    obs = embodiment.current
    assert tuple(obs.images) == ("front_rgb", "right_rgb", "wrist_rgb")
    assert set(obs.state) == {"eef_state"}
    assert obs.state["eef_state"].shape == (7,)
    assert set(obs.extra) == {"env_step"}
    for name, rgb in obs.images.items():
        assert rgb.shape == SHAPES[name] and rgb.dtype == np.uint8
        assert rgb.flags.c_contiguous and not rgb.flags.writeable
    assert embodiment.info.is_simulated
    assert embodiment.info.action_space.semantics.dim_labels == (
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "roll",
        "gripper",
    )
    assert embodiment.info.action_space.semantics.rotation_repr == "none"
    assert "blue_block" not in embodiment.info.docs
    assert "inside_target" not in repr(obs.extra)


def test_stale_bundle_rejected_after_execution_and_reset(embodiment):
    obs = embodiment.current
    old_images = {name: rgb.copy() for name, rgb in obs.images.items()}
    embodiment.begin_decision(obs)
    fresh = embodiment.execute_chunk(chunk([vector(embodiment)]), obs).observation
    assert fresh.state_time > obs.state_time
    assert (
        embodiment.observation_trace[-1]["generation"]
        > embodiment.observation_trace[-2]["generation"]
    )
    assert (
        embodiment.observation_trace[-1]["state_hash"]
        != embodiment.observation_trace[-2]["state_hash"]
    )
    with pytest.raises(S5Error, match="stale_observation"):
        embodiment.begin_decision(obs)
    for name in obs.images:
        np.testing.assert_array_equal(obs.images[name], old_images[name])
    embodiment.reset(Scene(id="again", instruction=S5A_INSTRUCTION))
    with pytest.raises(S5Error, match="stale_observation"):
        embodiment.begin_decision(fresh)


def test_cannot_consume_same_bundle_twice(embodiment):
    embodiment.begin_decision(embodiment.current)
    with pytest.raises(S5Error, match="already consumed"):
        embodiment.begin_decision(embodiment.current)


def test_out_of_band_state_change_invalidates_observation(embodiment):
    obs = embodiment.current
    embodiment.sim.hold(0.002)
    with pytest.raises(S5Error, match="stale_observation"):
        embodiment.begin_decision(obs)


def test_failed_render_cannot_reuse_previous_bundle(embodiment, monkeypatch):
    old = embodiment.current
    generation = embodiment.rig._generation
    draw = embodiment.rig._draw

    def fail(name):
        if name == "right_rgb":
            raise RuntimeError("injected renderer failure")
        return draw(name)

    monkeypatch.setattr(embodiment.rig, "_draw", fail)
    with pytest.raises(S5Error, match="observation_failure"):
        embodiment.observe()
    assert embodiment.rig._generation == generation
    assert embodiment.current is None
    with pytest.raises(S5Error, match="stale_observation"):
        embodiment.begin_decision(old)


@pytest.mark.parametrize(
    "change,category",
    [
        (lambda v: [*v, 0], "invalid_action"),
        (lambda v: [float("nan"), *v[1:]], "invalid_action"),
        (lambda v: [float("inf"), *v[1:]], "invalid_action"),
        (lambda v: [*v[:3], 0.1, *v[4:]], "invalid_action"),
        (lambda v: [*v[:4], -0.1, *v[5:]], "invalid_action"),
        (lambda v: [*v[:5], 0.1, v[6]], "invalid_action"),
        (lambda v: [*v[:6], -0.01], "invalid_action"),
        (lambda v: [*v[:6], 1.01], "invalid_action"),
        (lambda v: [*v[:6], 0], "safety_validation_abort"),
        (lambda v: [0.6, *v[1:]], "workspace_reject"),
        (lambda v: [v[0] + 0.03, *v[1:]], "invalid_action"),
    ],
)
def test_invalid_model_actions_rejected_before_any_motion(embodiment, change, category):
    initial_q = embodiment.sim.data.qpos.copy()
    initial_time = embodiment.sim.simulation_time_s
    embodiment.begin_decision(embodiment.current)
    with pytest.raises(S5Error, match=category):
        embodiment.execute_chunk(
            chunk([change(vector(embodiment))]), embodiment.current
        )
    np.testing.assert_array_equal(initial_q, embodiment.sim.data.qpos)
    assert embodiment.sim.simulation_time_s == initial_time
    assert embodiment.budget.steps == 0


def test_invalid_later_waypoint_does_not_execute_valid_prefix(embodiment):
    obs = embodiment.current
    valid = vector(embodiment)
    invalid = valid.copy()
    invalid[6] = -1
    embodiment.begin_decision(obs)
    with pytest.raises(S5Error, match="invalid_action"):
        embodiment.execute_chunk(chunk([valid, invalid]), obs)
    assert embodiment.sim.simulation_time_s == 0


def test_unreachable_target_rejected_by_isolated_ik_before_motion():
    limits = ActionLimits(chunk_path_m=2, episode_path_m=2, max_waypoints=64)
    env = NovaMujocoEmbodiment(Budget(RunLimits(max_episode_steps=64)), limits)
    try:
        obs = env.reset(Scene(id="unreachable", instruction="test"))
        before = env.sim.data.qpos.copy()
        start = env.sim.get_tool_pose().position_m
        path = np.linspace(start, [-0.64, 0.34, 0.44], 60)[1:]
        env.begin_decision(obs)
        with pytest.raises(S5Error, match="IK_fail"):
            env.execute_chunk(chunk([np.r_[p, 0, 0, 0, 1] for p in path]), obs)
        np.testing.assert_array_equal(before, env.sim.data.qpos)
        assert env.sim.simulation_time_s == 0
    finally:
        env.close()


def test_actionchunk_order_preserved(embodiment):
    obs = embodiment.current
    first = vector(embodiment)
    second = first.copy()
    second[0] += 0.002
    third = second.copy()
    third[2] -= 0.001
    actions = [first, second, third]
    embodiment.begin_decision(obs)
    result = embodiment.execute_chunk(chunk(actions), obs)
    assert not result.terminated
    trace = embodiment.action_trace[-1]
    np.testing.assert_array_equal(trace["ordered_actions"], actions)
    assert trace["executed_waypoints"] == 3
    np.testing.assert_allclose(
        result.observation.state["eef_state"][:3], third[:3], atol=1e-6
    )


def test_failed_runtime_waypoint_stops_remainder(embodiment, monkeypatch):
    obs = embodiment.current
    actions = [vector(embodiment, dz=dz) for dz in (0.002, 0.004, 0.006)]
    original = embodiment.executor.ik.solve
    calls = 0

    def injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        # Three whole-chunk preflight solves, then first actual waypoint succeeds.
        return replace(result, status="stagnated") if calls == 5 else result

    monkeypatch.setattr(embodiment.executor.ik, "solve", injected)
    embodiment.begin_decision(obs)
    with pytest.raises(S5Error, match="IK_fail"):
        embodiment.execute_chunk(chunk(actions), obs)
    trace = embodiment.action_trace[-1]
    assert trace["failed_index"] == 1
    assert trace["executed_waypoints"] == embodiment.budget.steps == 1
    assert calls == 5
    assert embodiment.current is None


def test_official_replay_loop_is_deterministic_without_openai():
    results = []
    for _ in range(2):
        env = NovaMujocoEmbodiment(Budget(RunLimits()))
        policy, transport = make_replay_policy(env, env.budget)
        try:
            result = run_loop(env, policy, transport)
            assert result["loop_pass"]
            assert result["checks"]["new_observation_received_by_next_decision"]
            assert result["provider_attempts"] == 2
            assert result["action_count"] >= 2
            assert not transport.live
            assert all(event["changed_images"] for event in result["events"])
            results.append(
                (env.action_trace, [o["state_hash"] for o in env.observation_trace])
            )
        finally:
            policy._client.close()
            env.close()
    assert results[0] == results[1]


def policy_with_reply(env, reply):
    transport = BoundedTransport(httpx.MockTransport(reply), env.budget, live=False)
    policy = agent_policy(
        model="offline-fixture",
        base_url="https://api.openai.com/v1",
        env={},
        wire="responses",
        wire_capture=False,
        images="always",
        depth="off",
        image_horizon=2,
        max_llm_calls=env.budget.limits.max_llm_calls,
        transport=transport,
        pre_check=env.pre_check,
    )
    policy.bind(env.info)
    return policy, transport


def response(tool, arguments):
    return httpx.Response(
        200,
        json={
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "fixture",
                    "name": tool,
                    "arguments": json.dumps(arguments),
                }
            ],
        },
    )


def test_actual_official_wire_contains_only_expected_observations(embodiment):
    captured = []

    def reply(request):
        captured.append(json.loads(request.content))
        obs = embodiment.current
        z = float(obs.state["eef_state"][2]) + 0.002
        return response(
            "move_to",
            {"targets": {"z": z, "gripper": 1}, "note": "Test bounded motion."},
        )

    policy, transport = policy_with_reply(embodiment, reply)
    try:
        result = run_loop(embodiment, policy, transport)
        assert result["loop_pass"]
        for payload in captured:
            assert payload["store"] is False
            assert {tool["name"] for tool in payload["tools"]} == {
                "move_to",
                "done",
                "give_up",
            }
            text = json.dumps(payload)
            for forbidden in (
                "qpos",
                "object_poses",
                "inside_target",
                "segmentation",
                "joint_pos",
                "top_rgb",
                "blue_block_collision",
                "camera_world_position",
            ):
                assert forbidden not in text
            latest = [
                item
                for item in payload["input"]
                if item.get("role") == "user" and isinstance(item.get("content"), list)
            ][-1]["content"]
            images = [
                part["image_url"] for part in latest if part["type"] == "input_image"
            ]
            assert len(images) == 3
            for data_url, shape in zip(images, SHAPES.values(), strict=True):
                image = np.asarray(
                    Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
                )
                assert image.shape == shape and image.dtype == np.uint8
        assert captured[0]["input"] != captured[1]["input"]
    finally:
        policy._client.close()


def test_official_tool_rejects_nonzero_orientation_and_call_budget(embodiment):
    def reply(_request):
        return response(
            "move_to", {"targets": {"yaw": 0.1}, "note": "Invalid orientation fixture."}
        )

    policy, transport = policy_with_reply(embodiment, reply)
    try:
        result = run_loop(embodiment, policy, transport)
        assert result["failure_category"] == "max_calls_reached"
        assert result["action_count"] == 0
        assert len(transport.events) == 2
        assert any(
            "fixed at 0" in m.get("content", "")
            for m in policy.transcript()
            if m["role"] == "tool"
        )
    finally:
        policy._client.close()


def test_model_done_is_not_success(embodiment):
    policy, transport = policy_with_reply(
        embodiment,
        lambda _req: response(
            "done", {"reason": "claimed success", "hindsight": "none"}
        ),
    )
    try:
        result = run_loop(embodiment, policy, transport)
        assert not result["loop_pass"]
        assert result["action_count"] == 0
        assert not result["evaluation_only"]["success"]
    finally:
        policy._client.close()


def test_missing_key_fails_closed_before_live_transport(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "HTTPTransport",
        lambda **_kw: pytest.fail("transport must not be constructed"),
    )
    with pytest.raises(S5Error, match="missing_api_key"):
        make_live_policy(
            None,
            Budget(RunLimits()),
            "openai/operator-verified-id",
            {"OPENROUTER_API_KEY": "alternate-not-allowed"},
            guard=SimulationIOGuard(),
        )


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "other/provider",
        "bare-model",
        "openai/model:free",
        "openai/<verified-id>",
    ],
)
def test_missing_or_wrong_model_does_not_invent_identifier(model):
    with pytest.raises(S5Error):
        live_settings(model, {"OPENAI_API_KEY": "offline-placeholder"})


def test_explicit_model_uses_official_environment_name():
    assert (
        live_settings(
            None,
            {
                "OPENAI_API_KEY": "offline-placeholder",
                "INSPECT_ROBOTS_MODEL": "openai/operator-verified-id",
            },
        )
        == "openai/operator-verified-id"
    )


def test_live_constructor_is_official_and_cannot_fall_back(embodiment, monkeypatch):
    def fail(_request):
        raise httpx.ReadTimeout("secret-must-not-be-logged")

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_kw: httpx.MockTransport(fail))
    policy, transport = make_live_policy(
        embodiment,
        embodiment.budget,
        "openai/operator-verified-id",
        {"OPENAI_API_KEY": "offline-placeholder", "OPENROUTER_API_KEY": "not-used"},
        guard=SimulationIOGuard(),
    )
    # Test only: prohibit network and inject the transport error directly.
    transport.guard = None
    try:
        assert policy.__class__.__module__ == "inspect_robots_agent.policy"
        assert policy.config.base_url == "https://api.openai.com/v1"
        assert policy.config.model == "operator-verified-id"
        result = run_loop(embodiment, policy, transport)
        assert result["failure_category"] == "model_timeout"
        assert result["action_count"] == 0
        assert len(transport.events) == 1
        assert "secret-must-not-be-logged" not in json.dumps(result)
    finally:
        policy._client.close()


def test_step_budget_rejects_whole_chunk_before_execution(embodiment):
    embodiment.budget.steps = embodiment.budget.limits.max_episode_steps - 1
    obs = embodiment.current
    embodiment.begin_decision(obs)
    with pytest.raises(S5Error, match="max_steps_reached"):
        embodiment.execute_chunk(
            chunk([vector(embodiment), vector(embodiment, dz=0.004)]), obs
        )
    assert embodiment.sim.simulation_time_s == 0


def test_actual_http_attempt_budget_and_error_redaction():
    calls = []

    def reply(_request):
        calls.append(1)
        return httpx.Response(500, json={"error": "secret-do-not-log"})

    budget = Budget(RunLimits(max_llm_calls=1))
    transport = BoundedTransport(httpx.MockTransport(reply), budget, live=False)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    with pytest.raises(S5Error) as error:
        transport.handle_request(request)
    assert error.value.category == "provider/API_error"
    assert "secret-do-not-log" not in str(error.value)
    with pytest.raises(S5Error, match="max_calls_reached"):
        transport.handle_request(request)
    assert calls == [1]
    assert "secret-do-not-log" not in json.dumps(transport.events)


def test_wall_deadline_interrupts_blocking_work():
    started = time.monotonic()
    with pytest.raises(S5Error, match="max_wall_time_reached"), wall_deadline(0.02):
        time.sleep(1)
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(
    "event,args",
    [
        ("socket.connect", (None, ("172.10.10.100", 29999))),
        ("socket.connect", (None, ("172.10.10.100", 30004))),
        ("socket.getaddrinfo", ("172.10.10.100", 29999)),
        ("open", ("/dev/ttyACM0", "rb", 0)),
        ("open", ("/dev/video0", "rb", 0)),
    ],
)
def test_hardware_guard_rejects_before_io(event, args):
    guard = SimulationIOGuard()
    guard.active = True  # Hardware remains forbidden even during provider calls.
    with pytest.raises(S5Error, match="prohibited_io"):
        guard.check(event, args)


def test_reviewed_official_source_is_unmodified():
    pin = verify_official_stack()
    assert pin["revision"] == "7e506e3b6f39530f18e75cf97b23c4287aa90f12"


@pytest.mark.parametrize(
    "key,model,category",
    [
        (None, None, "missing_api_key"),
        ("s5-synthetic-secret", None, "missing_model_identifier"),
        ("s5-synthetic-secret", "openai/operator-verified-id", "quality_gate_failure"),
    ],
)
def test_live_cli_preflight_is_closed_and_evidence_is_secret_free(
    tmp_path, key, model, category
):
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env.pop("INSPECT_ROBOTS_MODEL", None)
    if key:
        env["OPENAI_API_KEY"] = key
    output = tmp_path / "blocked"
    command = [sys.executable, "run_s5a.py", "--mode", "live", "--output", str(output)]
    if model:
        command += ["--model", model]
    result = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    summary = json.loads(result.stdout)
    assert summary["status"] == "BLOCKED"
    assert summary["failure_category"] == category
    assert all(value == 0 for value in summary["side_effects"].values())
    for name in (
        "validation.json",
        "live_agent_result.json",
        "observation_trace.jsonl",
        "action_trace.jsonl",
        "timing.json",
        "environment.json",
        "quality_gates.json",
    ):
        assert (output / name).is_file()
        if key:
            assert key not in (output / name).read_text()
    assert not (output / "action_trace.jsonl").read_text()
    assert not result.stderr


def test_transport_will_not_record_key_echo_in_request_id():
    secret = "syntheticcredential"
    transport = BoundedTransport(
        httpx.MockTransport(
            lambda _r: httpx.Response(
                200, headers={"x-request-id": "req_" + secret}, json={"output": []}
            )
        ),
        Budget(RunLimits()),
        live=False,
    )
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer " + secret},
    )
    transport.handle_request(request)
    assert secret not in json.dumps(transport.events)


def test_transport_rejects_other_destinations_before_attempt():
    transport = BoundedTransport(
        httpx.MockTransport(lambda _r: pytest.fail("must not send")),
        Budget(RunLimits()),
        live=False,
    )
    with pytest.raises(S5Error, match="unexpected provider destination"):
        transport.handle_request(
            httpx.Request("POST", "https://example.invalid/responses")
        )
    assert transport.budget.requests == 0


def test_failed_post_motion_render_does_not_label_stale_eef_as_final(
    embodiment, monkeypatch
):
    original = embodiment.rig.observe
    renders = 0

    def fail_after_motion(**kwargs):
        nonlocal renders
        renders += 1
        if renders > 1:
            raise RuntimeError("injected post-motion render failure")
        return original(**kwargs)

    monkeypatch.setattr(embodiment.rig, "observe", fail_after_motion)
    policy, transport = make_replay_policy(embodiment, embodiment.budget)
    try:
        result = run_loop(embodiment, policy, transport)
        assert not result["loop_pass"]
        assert result["failure_category"] == "observation_failure"
        assert result["final_eef_state"] is None
        assert result["action_count"] > 0
        assert result["last_observation_eef_state"] == result["initial_eef_state"]
    finally:
        policy._client.close()
