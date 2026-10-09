#!/usr/bin/env python3
"""S5A only. Explicit live/replay modes, fail-closed credentials, no S5B launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from s5_common import (
    INTEGRATED_HEAD,
    BENCHMARK_INSTRUCTION,
    ROOT,
    S5A_INSTRUCTION,
    Budget,
    RunLimits,
    S5Error,
    live_settings,
    install_log_redaction,
    parse_env_file,
    redact_value,
    register_credentials,
    provider_profile,
    runtime_versions,
    verify_official_stack,
    wall_deadline,
)


def source_manifest() -> dict:
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "configs").glob("*.json"))
    paths += sorted((ROOT / "scenes/benchmark_v0").glob("*.xml"))
    paths += [ROOT / "nova2.xml", ROOT / "pytest.ini"]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def json_write(path: Path, value, *, lines=False):
    # Final defense, without ever printing or writing the credential itself.
    value = redact_value(value)
    text = (
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in value)
        if lines
        else json.dumps(value, indent=2, allow_nan=False) + "\n"
    )
    path.write_text(text)


def quality_result(path: Path | None) -> dict:
    if path is None:
        return {
            "status": "NOT_RUN",
            "reason": "no current quality-gate record supplied",
        }
    record = json.loads(path.read_text())
    required = {
        "pytest",
        "ruff",
        "ruff_format",
        "mypy",
        "compile",
        "s1_s2_benchmark",
        "diff_check",
    }
    checks = record.get("checks", {})
    if (
        record.get("source_manifest") != source_manifest()
        or record.get("runtime_versions") != runtime_versions()
        or set(checks) != required
        or any(row.get("exit_code") != 0 for row in checks.values())
        or record.get("status") != "PASS"
    ):
        raise S5Error(
            "quality_gate_failure", "quality record is incomplete, failed or stale"
        )
    return record


def main(argv=None, *, benchmark: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("live", "replay"))
    parser.add_argument(
        "--model", help="Explicit Astra ID; bare with --base-url, openai/ID otherwise"
    )
    parser.add_argument(
        "--base-url", help="Explicit operator-approved HTTPS Responses base"
    )
    parser.add_argument(
        "--provider-env-file",
        type=Path,
        help="Literal credential env file; never sourced or executed",
    )
    parser.add_argument(
        "--provider-proxy",
        help="Explicit HTTPS provider proxy URL; ambient proxy variables are ignored",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-llm-calls", type=int, default=12 if benchmark else 2)
    parser.add_argument(
        "--max-episode-steps", type=int, default=384 if benchmark else 32
    )
    parser.add_argument("--max-decisions", type=int, default=12 if benchmark else 2)
    parser.add_argument(
        "--max-wall-time", type=float, default=600.0 if benchmark else 90.0
    )
    parser.add_argument("--quality-gates", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runtime_env = dict(os.environ)
    register_credentials(runtime_env)
    install_log_redaction()
    output = args.output or ROOT / (
        "evidence/sim_s5b" if benchmark else "evidence/sim_s5a"
    ) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode)
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    environment = {
        "repository_head": head,
        "integrated_base": INTEGRATED_HEAD,
        "python": platform.python_version(),
        "mode": args.mode,
        "stage": "S5B" if benchmark else "S5A",
        "model_identifier_used": None,
        "source_manifest": source_manifest(),
        "scene": "benchmark_v0",
        "camera_profile": "configs/s3_s4_cameras.json",
        "requested_provider_configuration": {
            "provider": "openai",
            "wire": "responses",
            "effort": args.effort,
            "images": "always",
            "depth": "off",
            "image_horizon": 2,
            "model": "explicit operator-verified identifier required",
        },
        "api_key_present": False,
        "model_identifier_configured": bool(
            args.model or runtime_env.get("INSPECT_ROBOTS_MODEL")
        ),
    }
    result: dict = {"status": "BLOCKED", "loop_pass": False}
    quality = {"status": "NOT_RUN"}
    observations: list[dict] = []
    actions: list[dict] = []
    events: list[dict] = []
    guard = embodiment = policy = transport = budget = None
    try:
        if benchmark:
            from s5_benchmark import S5A_HEAD, require_s5a_pass

            require_s5a_pass()
            if subprocess.run(
                ["git", "merge-base", "--is-ancestor", S5A_HEAD, "HEAD"],
                cwd=ROOT,
                check=False,
            ).returncode:
                raise S5Error(
                    "precondition_failure", "S5A checkpoint must be an ancestor"
                )
        if args.provider_env_file is not None:
            runtime_env.update(
                parse_env_file(args.provider_env_file, allowed_names={"RELAY_API_KEY"})
            )
            register_credentials(runtime_env)
        if args.provider_proxy:
            runtime_env["S5_PROVIDER_HTTPS_PROXY"] = args.provider_proxy
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", INTEGRATED_HEAD, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise S5Error(
                "precondition_failure", "integrated checkpoint is not an ancestor"
            )
        checkpoint = json.loads(
            (ROOT / "evidence/sim_s3_s4/20260927T162753Z/validation.json").read_text()
        )
        if checkpoint.get("status") != "PASS":
            raise S5Error(
                "precondition_failure", "S3/S4 integration evidence is not PASS"
            )
        limits: RunLimits
        if benchmark:
            from s5_common import BenchmarkRunLimits

            limits = BenchmarkRunLimits(
                args.max_llm_calls,
                args.max_episode_steps,
                args.max_decisions,
                args.max_wall_time,
            )
        else:
            limits = RunLimits(
                args.max_llm_calls,
                args.max_episode_steps,
                args.max_decisions,
                args.max_wall_time,
            )
        environment["limits"] = asdict(limits)
        quality = quality_result(args.quality_gates)
        # Credential/model gate precedes imports, simulator construction and I/O.
        model = (
            live_settings(args.model, runtime_env, base_url=args.base_url)
            if args.mode == "live"
            else None
        )
        if args.mode == "live":
            profile = provider_profile(args.base_url, runtime_env)
            environment["provider_profile"] = {
                "base_url": profile["base_url"],
                "api_key_env": profile["api_key_env"],
                "explicit_proxy_configured": bool(profile["proxy"]),
                "transport_mode": "explicit_proxy" if profile["proxy"] else "direct",
                "tls_verification": True,
                "redirects_allowed": False,
            }
            environment["api_key_present"] = True
        if args.mode == "live" and quality["status"] != "PASS":
            raise S5Error(
                "quality_gate_failure",
                "live S5 requires passing current regression checks",
            )
        environment["official_stack"] = verify_official_stack()
        environment["runtime_versions"] = runtime_versions()
        import mujoco
        import numpy as np

        from s5_agent import SimulationIOGuard, make_live_policy, make_replay_policy
        from s5_embodiment import NovaMujocoEmbodiment
        from s5_loop import run_loop

        environment.update({"mujoco": mujoco.__version__, "numpy": np.__version__})
        guard = SimulationIOGuard(
            args.base_url if args.mode == "live" else None,
            proxy=profile["proxy"] if args.mode == "live" else None,
        )
        guard.install()
        budget = Budget(limits)
        with wall_deadline(limits.max_wall_time):
            if benchmark:
                from s5_benchmark import benchmark_action_limits

                embodiment = NovaMujocoEmbodiment(budget, benchmark_action_limits())
            else:
                embodiment = NovaMujocoEmbodiment(budget)
            if args.mode == "live":
                policy, transport = make_live_policy(
                    embodiment,
                    budget,
                    model,
                    runtime_env,
                    effort=args.effort,
                    guard=guard,
                    base_url=args.base_url,
                )
            else:
                policy, transport = make_replay_policy(embodiment, budget)
            environment["policy_configuration"] = asdict(policy.config)
            environment["model_identifier_used"] = policy.config.model
            environment["renderer"] = embodiment.rig.renderer_info
            environment["action_limits"] = asdict(embodiment.limits)
            benchmark_recorder = None
            if benchmark:
                from s5_benchmark import BenchmarkRecorder

                benchmark_recorder = BenchmarkRecorder(output)
            result = run_loop(
                embodiment, policy, transport, benchmark=benchmark_recorder
            )
            live_request = args.mode == "live" and any(
                e["status"] == 200 for e in transport.events
            )
            result["live_request_performed"] = live_request
            result["status"] = (
                "PASS"
                if result["loop_pass"] and live_request and quality["status"] == "PASS"
                else "OFFLINE_LOOP_PASS"
                if result["loop_pass"] and args.mode == "replay"
                else "PARTIAL"
            )
            if benchmark:
                result["status"] = (
                    "PASS"
                    if result.get("benchmark_success")
                    and live_request
                    and quality["status"] == "PASS"
                    else "PARTIAL"
                )
            # Human review only; these images are the exact policy RGB frames.
            if embodiment.current is not None:
                from PIL import Image

                for name, rgb in embodiment.current.images.items():
                    Image.fromarray(rgb).save(output / f"after_{name}.png")
    except S5Error as exc:
        result.update(
            {
                "status": "BLOCKED" if budget is None else "PARTIAL",
                "failure_category": exc.category,
                "failure_detail": exc.detail,
            }
        )
    except Exception:  # noqa: BLE001 - never include arbitrary provider/configuration errors
        result.update(
            {
                "status": "BLOCKED" if budget is None else "PARTIAL",
                "failure_category": "runtime_error",
                "failure_detail": "initialization or execution failed; no raw exception recorded",
            }
        )
    finally:
        if embodiment is not None:
            observations = embodiment.observation_trace
            actions = embodiment.action_trace
            events = result.get("events", [])
            result["validation_results"] = embodiment.validation_trace
            if "timing" not in result:
                result["timing"] = {"partial": True, **embodiment.timing}
            try:
                embodiment.close()
            except Exception:  # noqa: BLE001 - preserve evidence without raw error text
                result.update(
                    {"status": "PARTIAL", "failure_category": "cleanup_failure"}
                )
        if policy is not None:
            # This pin has no public close(); the owned ResponsesClient has one.
            try:
                policy._client.close()
            except Exception:  # noqa: BLE001 - provider cleanup may include sensitive text
                result.update(
                    {"status": "PARTIAL", "failure_category": "cleanup_failure"}
                )
        elif transport is not None:
            transport.close()
    side_effects = dict.fromkeys(
        (
            "robot_connections",
            "robot_commands",
            "serial_connections",
            "physical_camera_connections",
            "openai_requests",
        ),
        0,
    )
    if transport is not None and transport.live:
        side_effects["openai_requests"] = transport.transmitted_requests
        if result.get("failure_category") == "provider_network_configuration":
            result["status"] = "BLOCKED"
    result.update(
        {
            "mode": args.mode,
            "instruction": BENCHMARK_INSTRUCTION if benchmark else S5A_INSTRUCTION,
            "side_effects": side_effects,
            "S5B": "BENCHMARK_ATTEMPTED"
            if benchmark
            else "NOT_STARTED; this runner is S5A only",
        }
    )
    result["provider_attempts"] = 0 if budget is None else budget.requests
    result["transmitted_requests"] = (
        0 if transport is None else transport.transmitted_requests
    )
    result["provider_events"] = [] if transport is None else transport.events
    result["blocked_io_attempts"] = {} if guard is None else guard.blocked
    json_write(output / "environment.json", environment)
    json_write(output / "quality_gates.json", quality)
    json_write(output / "live_agent_result.json", result)
    json_write(
        output / "validation.json",
        {
            "status": result["status"],
            "live_s5b_pass" if benchmark else "live_s5a_pass": result["status"]
            == "PASS",
            "checks": result.get("checks", {}),
            "failure_category": result.get("failure_category"),
            "side_effects": side_effects,
        },
    )
    json_write(output / "timing.json", result.get("timing", {"status": "NOT_MEASURED"}))
    json_write(output / "observation_trace.jsonl", observations, lines=True)
    json_write(output / "action_trace.jsonl", actions, lines=True)
    json_write(output / "episode_trace.jsonl", events, lines=True)
    if benchmark:
        json_write(output / "episode_result.json", result)
        json_write(output / "evaluator_metrics.json", result.get("evaluation_only", {}))
        json_write(
            output / "evaluator_trace.jsonl",
            result.get("evaluator_trace", []),
            lines=True,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "failure_category": result.get("failure_category"),
                "evidence": str(output),
                "side_effects": side_effects,
            }
        )
    )
    return 0 if result["status"] in {"PASS", "OFFLINE_LOOP_PASS"} else 2


if __name__ == "__main__":
    sys.exit(main())
