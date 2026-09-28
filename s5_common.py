"""S5A configuration, budgets and reviewed official-source provenance."""

from __future__ import annotations

import hashlib
import importlib.util
from importlib.metadata import version
import json
import math
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTEGRATED_HEAD = "2bf4650d08ae3812836c3562defc16976a27db70"
OFFICIAL_REVISION = "7e506e3b6f39530f18e75cf97b23c4287aa90f12"
YAM_REVISION = "178c93037afcf593618e0a1db6ae1f1b0cfeaa0f"
BENCHMARK_INSTRUCTION = "Pick up the blue block and place it inside the red box."
S5A_INSTRUCTION = (
    "Simulation-only action-loop check. From the initial observed EEF position, "
    "use move_to to move the nominal grasp center 0.005 metres upward in base Z "
    "with gripper fully open (1). After receiving the fresh post-motion images "
    "and EEF state, use move_to to move 0.005 metres downward with gripper 1. "
    "Keep X and Y unchanged and yaw, pitch and roll zero. Do not grasp objects. "
    "Use only the observations; no simulator internals are available."
)


class S5Error(RuntimeError):
    """A fixed, secret-free reason for a closed-loop stop."""

    def __init__(self, category: str, detail: str):
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


@dataclass(frozen=True)
class RunLimits:
    max_llm_calls: int = 2
    max_episode_steps: int = 32  # Executed waypoints, not model decisions.
    max_decisions: int = 2
    max_wall_time: float = 90.0

    def __post_init__(self):
        for name, ceiling in (
            ("max_llm_calls", 3),
            ("max_episode_steps", 64),
            ("max_decisions", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"S5A {name} must be an integer in [1,{ceiling}]")
        if (
            isinstance(self.max_wall_time, bool)
            or not math.isfinite(self.max_wall_time)
            or not 0 < self.max_wall_time <= 300
        ):
            raise ValueError("S5A max_wall_time must be finite in (0,300]")


class Budget:
    def __init__(self, limits: RunLimits):
        self.limits = limits
        self.started = time.monotonic()
        self.deadline = self.started + limits.max_wall_time
        self.requests = 0
        self.steps = 0

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise S5Error("max_wall_time_reached", "episode wall-time budget exhausted")
        return remaining

    def request(self) -> None:
        self.remaining()
        if self.requests >= self.limits.max_llm_calls:
            raise S5Error("max_calls_reached", "request budget exhausted")
        self.requests += 1

    def check_steps(self, count: int) -> None:
        self.remaining()
        if self.steps + count > self.limits.max_episode_steps:
            raise S5Error("max_steps_reached", "entire chunk exceeds remaining steps")


@contextmanager
def wall_deadline(seconds: float):
    """Interrupt blocking model/render work on the single-threaded Linux CLI."""

    def expired(_signum, _frame):
        raise S5Error("max_wall_time_reached", "episode wall-time budget exhausted")

    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def source_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def verify_official_stack() -> dict:
    """Refuse accidental imports from the dirty historical reference checkout."""
    pin = json.loads((ROOT / "configs/s5_official_stack.json").read_text())
    for name, expected in pin["packages"].items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise S5Error("official_stack_mismatch", "official source path unavailable")
        actual = source_digest(Path(spec.origin).resolve().parent)
        if actual != expected["python_tree_sha256"]:
            raise S5Error(
                "official_stack_mismatch", "installed source differs from reviewed pin"
            )
    return pin


def runtime_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in (
            "mujoco",
            "numpy",
            "httpx",
            "httpcore",
            "websockets",
            "Pillow",
            "PyOpenGL",
            "pytest",
            "ruff",
            "mypy",
        )
    }


def live_settings(model: str | None, env: dict[str, str]) -> str:
    """Only explicit operator configuration; never a model/key/provider fallback."""
    if not env.get("OPENAI_API_KEY", "").strip():
        raise S5Error("missing_api_key", "OPENAI_API_KEY must be set by the operator")
    requested = model or env.get("INSPECT_ROBOTS_MODEL")
    if not requested:
        raise S5Error(
            "missing_model_identifier",
            "supply --model openai/<verified Astra API id> or INSPECT_ROBOTS_MODEL",
        )
    # This validates routing syntax, not model availability or family membership.
    # The operator supplies the verified Astra identifier; no alias is invented.
    if (
        not requested.startswith("openai/")
        or not requested.removeprefix("openai/")
        or any(c.isspace() for c in requested)
        or any(c in requested for c in (":", "?", "#", "\\"))
        or requested.count("/") != 1
        or "<" in requested
    ):
        raise S5Error("invalid_model_configuration", "explicit openai/model required")
    return requested
