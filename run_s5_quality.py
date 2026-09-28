#!/usr/bin/env python3
"""Run offline regression gates and bind their results to exact local source bytes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_s5a import json_write, source_manifest
from s5_common import ROOT, runtime_versions, verify_official_stack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    before = source_manifest()
    official = verify_official_stack()
    python = sys.executable
    modules = [
        "s5_common.py",
        "s5_agent.py",
        "s5_embodiment.py",
        "s5_loop.py",
        "run_s5a.py",
        "run_s5_quality.py",
    ]
    commands = {
        "pytest": [python, "-m", "pytest", "-q"],
        "ruff": [
            python,
            "-m",
            "ruff",
            "check",
            "--exclude",
            "DOBOT_6Axis_ROS2_V4",
            ".",
        ],
        "ruff_format": [
            python,
            "-m",
            "ruff",
            "format",
            "--check",
            "--exclude",
            "DOBOT_6Axis_ROS2_V4",
            ".",
        ],
        "mypy": [
            python,
            "-m",
            "mypy",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            "--check-untyped-defs",
            *modules,
        ],
        "compile": [
            python,
            "-m",
            "py_compile",
            *[p.name for p in sorted(ROOT.glob("*.py"))],
        ],
        "s1_s2_benchmark": [
            python,
            "run_s1_s2_benchmark.py",
            "--output",
            str(output / "s1_s2_benchmark"),
        ],
        "diff_check": ["git", "diff", "--check"],
    }
    # Offline subprocesses do not inherit credentials, regardless of the shell.
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(
            word in key.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        )
    }
    env["MUJOCO_GL"] = "egl"
    checks = {}
    for name, command in commands.items():
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
        )
        checks[name] = {
            "command": command,
            "exit_code": completed.returncode,
            "elapsed_s": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        print(f"{name}: {'PASS' if completed.returncode == 0 else 'FAIL'}", flush=True)
    passed = (
        all(row["exit_code"] == 0 for row in checks.values())
        and before == source_manifest()
    )
    record = {
        "status": "PASS" if passed else "FAIL",
        "source_manifest": before,
        "runtime_versions": runtime_versions(),
        "official_stack": official,
        "checks": checks,
        "mypy_scope": "S5 adapter/runner bodies; imported frozen S1-S4 and official packages skipped",
    }
    json_write(output / "quality_gates.json", record)
    print(
        json.dumps(
            {"status": record["status"], "record": str(output / "quality_gates.json")}
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
