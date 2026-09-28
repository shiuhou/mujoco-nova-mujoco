"""Bounded S5A loop over the official policy lifecycle and simulation adapter."""

from __future__ import annotations

import time

import numpy as np
from inspect_robots.scene import Scene

from s5_agent import observable_tools
from s5_common import S5A_INSTRUCTION, S5Error


def run_loop(embodiment, policy, transport) -> dict:
    budget = embodiment.budget
    scene = Scene(id="s5a_bounded_vertical_check", instruction=S5A_INSTRUCTION)
    observation = embodiment.reset(scene)
    initial_eef = observation.state["eef_state"].copy()
    policy.reset(scene)
    events = []
    timing = dict.fromkeys(
        ("T_encode_or_request_prep", "T_model", "T_policy_parse", "T_total_step"), 0.0
    )
    failure = None
    failure_detail = None
    decisions = 0
    initial = {
        "robot_q_rad": embodiment.sim.get_joint_position().tolist(),
        "object_poses_evaluator_only": {
            obj.semantic_name: embodiment.sim.get_object_state(obj.semantic_name)
            for obj in embodiment.sim.scene_spec.objects
        },
    }
    for decision in range(budget.limits.max_decisions):
        started = time.perf_counter()
        input_generation = embodiment.observation_trace[-1]["generation"]
        event = {"decision": decision, "input_generation": input_generation}
        try:
            budget.remaining()
            if budget.requests >= budget.limits.max_llm_calls:
                raise S5Error("max_calls_reached", "request budget exhausted")
            embodiment.begin_decision(observation)
            transport.turn_first_request = None
            model_before = transport.seconds
            before = dict(embodiment.timing)
            act_started = time.perf_counter()
            try:
                chunk = policy.act(observation)
            finally:
                act_elapsed = time.perf_counter() - act_started
                prep = (
                    max(0, transport.turn_first_request - act_started)
                    if transport.turn_first_request is not None
                    else 0.0
                )
                model_elapsed = transport.seconds - model_before
                validation = sum(
                    embodiment.timing[key] - before[key]
                    for key in ("T_IK", "T_chunk_validation")
                )
                timing["T_encode_or_request_prep"] += prep
                timing["T_model"] += model_elapsed
                # Public act() has no parse hook. This measured residual includes
                # official tool parsing, interpolation, internal re-prompt prep.
                timing["T_policy_parse"] += max(
                    0, act_elapsed - prep - model_elapsed - validation
                )
            decisions += 1
            if any(action.meta.get("request_stop") for action in chunk.actions):
                reason = chunk.actions[0].meta.get("stop_detail")
                if reason == "LLM call budget exhausted":
                    raise S5Error(
                        "max_calls_reached", "official policy exhausted its call budget"
                    )
                raise S5Error(
                    "policy_stopped", "terminal model output did not execute motion"
                )
            step = embodiment.execute_chunk(chunk, observation)
            fresh = step.observation
            previous_meta, next_meta = embodiment.observation_trace[-2:]
            actual_displacement = float(
                np.linalg.norm(
                    fresh.state["eef_state"][:3] - observation.state["eef_state"][:3]
                )
            )
            event.update(
                {
                    "output_generation": next_meta["generation"],
                    "fresh_after_settle": (
                        next_meta["generation"] > input_generation
                        and fresh.state_time > observation.state_time
                        and next_meta["state_hash"] != previous_meta["state_hash"]
                    ),
                    "eef_displacement_m": actual_displacement,
                    "changed_images": [
                        name
                        for name in fresh.images
                        if next_meta["images"][name]["sha256"]
                        != previous_meta["images"][name]["sha256"]
                    ],
                    "waypoints": len(chunk.actions),
                }
            )
            if not event["fresh_after_settle"]:
                raise S5Error(
                    "stale_observation", "post-settle freshness was not established"
                )
            observation = fresh
        except S5Error as exc:
            failure, failure_detail = exc.category, exc.detail
            event["failure"] = failure
        except Exception:  # noqa: BLE001 - raw provider/model exception text is never evidence
            failure = (
                "invalid_action"
                if transport.events and transport.events[-1]["status"] == 200
                else "provider/API_error"
            )
            failure_detail = "official policy or embodiment rejected the operation"
            event["failure"] = failure
        finally:
            event["T_total_step"] = time.perf_counter() - started
            timing["T_total_step"] += event["T_total_step"]
            events.append(event)
        if failure:
            break
    motion = any(event.get("eef_displacement_m", 0) > 1e-5 for event in events)
    fresh = bool(events) and all(
        event.get("fresh_after_settle", False) for event in events
    )
    executed = budget.steps > 0 and all(
        row["executor_status"] == "completed" for row in embodiment.action_trace
    )
    tools = observable_tools(policy)
    checks = {
        "official_agent_used": policy.__class__.__module__
        == "inspect_robots_agent.policy",
        "generic_move_to_used": any(row["tool"] == "move_to" for row in tools),
        "actionchunk_executed": executed,
        "simulation_eef_changed_through_execution": motion,
        "settle_occurred": embodiment.timing["T_settle"] > 0,
        "fresh_post_motion_observation": fresh,
        "new_observation_received_by_next_decision": len(events) > 1
        and events[1]["input_generation"] == events[0].get("output_generation"),
        "no_policy_privileged_state": set(observation.state) == {"eef_state"}
        and set(observation.extra) == {"env_step"},
        "no_abort": failure is None,
    }
    # One successful motion suffices for the minimal S5A contract. The default
    # two-decision run also records continuation using the post-settle bundle.
    required = {
        k: v
        for k, v in checks.items()
        if k != "new_observation_received_by_next_decision"
    }
    return {
        "loop_pass": all(required.values()),
        "checks": checks,
        "failure_category": failure,
        "failure_detail": failure_detail,
        "decisions": decisions,
        "provider_attempts": budget.requests,
        "action_count": budget.steps,
        "path_length_m": embodiment.path_m,
        "initial": initial,
        "initial_eef_state": initial_eef.tolist(),
        "last_observation_eef_state": observation.state["eef_state"].tolist(),
        "final_eef_state": (
            None
            if embodiment.current is None
            else embodiment.current.state["eef_state"].tolist()
        ),
        "episode_duration_s": time.monotonic() - budget.started,
        "simulation_time_s": embodiment.sim.simulation_time_s,
        "timing": {**timing, **embodiment.timing},
        "events": events,
        "tools": tools,
        "evaluation_only": embodiment.sim.compute_manipulation_metrics(),
    }
