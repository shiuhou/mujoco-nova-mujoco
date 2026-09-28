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
    ROOT,
    S5A_INSTRUCTION,
    Budget,
    RunLimits,
    S5Error,
    live_settings,
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
    text = (
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in value)
        if lines
        else json.dumps(value, indent=2, allow_nan=False) + "\n"
    )
    secret = os.environ.get("OPENAI_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED]")
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("live", "replay"))
    parser.add_argument(
        "--model", help="Explicit Astra ID; bare with --base-url, openai/ID otherwise"
    )
    parser.add_argument(
        "--base-url", help="Explicit operator-approved HTTPS Responses base"
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-llm-calls", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=32)
    parser.add_argument("--max-decisions", type=int, default=2)
    parser.add_argument("--max-wall-time", type=float, default=90.0)
    parser.add_argument("--quality-gates", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "evidence/sim_s5a" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode
    )
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    environment = {
        "repository_head": head,
        "integrated_base": INTEGRATED_HEAD,
        "python": platform.python_version(),
        "mode": args.mode,
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
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "model_identifier_configured": bool(
            args.model or os.environ.get("INSPECT_ROBOTS_MODEL")
        ),
    }
    result: dict = {"status": "BLOCKED", "loop_pass": False}
    quality = {"status": "NOT_RUN"}
    observations: list[dict] = []
    actions: list[dict] = []
    events: list[dict] = []
    guard = embodiment = policy = transport = budget = None
    try:
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
            live_settings(args.model, dict(os.environ), base_url=args.base_url)
            if args.mode == "live"
            else None
        )
        if args.mode == "live" and quality["status"] != "PASS":
            raise S5Error(
                "quality_gate_failure",
                "live S5A requires passing current regression checks",
            )
        environment["official_stack"] = verify_official_stack()
        environment["runtime_versions"] = runtime_versions()
        import mujoco
        import numpy as np

        from s5_agent import SimulationIOGuard, make_live_policy, make_replay_policy
        from s5_embodiment import NovaMujocoEmbodiment
        from s5_loop import run_loop

        environment.update({"mujoco": mujoco.__version__, "numpy": np.__version__})
        guard = SimulationIOGuard(args.base_url if args.mode == "live" else None)
        guard.install()
        budget = Budget(limits)
        with wall_deadline(limits.max_wall_time):
            embodiment = NovaMujocoEmbodiment(budget)
            if args.mode == "live":
                policy, transport = make_live_policy(
                    embodiment,
                    budget,
                    model,
                    dict(os.environ),
                    effort=args.effort,
                    guard=guard,
                    base_url=args.base_url,
                )
            else:
                policy, transport = make_replay_policy(embodiment, budget)
            environment["policy_configuration"] = asdict(policy.config)
            environment["model_identifier_used"] = policy.config.model
            environment["renderer"] = embodiment.rig.renderer_info
            result = run_loop(embodiment, policy, transport)
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
        side_effects["openai_requests"] = len(transport.events)
    result.update(
        {
            "mode": args.mode,
            "instruction": S5A_INSTRUCTION,
            "side_effects": side_effects,
            "S5B": "BLOCKED; not implemented or launched before live S5A PASS",
        }
    )
    result["provider_attempts"] = 0 if budget is None else budget.requests
    result["provider_events"] = [] if transport is None else transport.events
    result["blocked_io_attempts"] = {} if guard is None else guard.blocked
    json_write(output / "environment.json", environment)
    json_write(output / "quality_gates.json", quality)
    json_write(output / "live_agent_result.json", result)
    json_write(
        output / "validation.json",
        {
            "status": result["status"],
            "live_s5a_pass": result["status"] == "PASS",
            "checks": result.get("checks", {}),
            "failure_category": result.get("failure_category"),
            "side_effects": side_effects,
        },
    )
    json_write(output / "timing.json", result.get("timing", {"status": "NOT_MEASURED"}))
    json_write(output / "observation_trace.jsonl", observations, lines=True)
    json_write(output / "action_trace.jsonl", actions, lines=True)
    json_write(output / "episode_trace.jsonl", events, lines=True)
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
