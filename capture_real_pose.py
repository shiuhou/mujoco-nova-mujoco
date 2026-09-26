#!/usr/bin/env python3
"""Capture read-only Dobot Nova2 kinematic samples.

This utility talks to the Dashboard port only to query the four explicitly
allow-listed state/kinematic commands.  It never requests control, changes
robot state, or sends a motion command.  Captured records are intended for
offline comparison with :mod:`compare_pose`.

The Dashboard wire format is ASCII and terminates each response with ``;``::

    ErrorID,{value,...},Command(...);

The JSON metadata records the empirically observed Dobot Rx/Ry/Rz convention:
XYZ roll/pitch/yaw (``rpy``), with angles in degrees.  The comparison still
uses rotation matrices and never subtracts the three components directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


DEFAULT_HOST = "172.10.10.100"
DEFAULT_PORT = 29999
USER = 0
TOOL = 0
# Twelve read-only Nova2 samples matched this convention to <0.001 deg.
ORIENTATION_CONVENTION = "rpy"
SOURCE = "dobot_tcp_29999_read_only"

# Keep this tuple deliberately closed.  DashboardClient.query rejects every
# string other than these exact commands, including commands typed by a user.
ALLOWED_COMMANDS = (
    "RobotMode()",
    "GetErrorID()",
    "GetAngle()",
    "GetPose(user=0,tool=0)",
)
ALLOWED_COMMAND_SET = frozenset(ALLOWED_COMMANDS)

ROBOT_MODE_NAMES = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWEROFF",
    4: "DISABLED",
    5: "ENABLE_IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "SINGLE_MOVE",
    9: "ERROR",
    10: "PAUSE",
    11: "COLLISION",
}

# These states can change while the query sequence is being collected, or
# indicate an active alarm.  Saving a normal paired sample would be unsafe or
# kinematically ambiguous.
UNRELIABLE_ROBOT_MODES = frozenset({1, 2, 7, 8, 9, 10, 11})
READABLE_ROBOT_MODES = frozenset({3, 4, 5, 6})


class SocketLike(Protocol):
    def sendall(self, data: bytes) -> None: ...

    def recv(self, size: int) -> bytes: ...

    def settimeout(self, timeout: float | None) -> None: ...

    def close(self) -> None: ...


class DashboardProtocolError(ValueError):
    """Raised when a Dashboard response is malformed or mismatched."""


class CaptureError(RuntimeError):
    """Raised when a snapshot cannot be saved as a normal sample."""


@dataclass(frozen=True)
class DashboardResponse:
    error_id: int
    values: tuple[str, ...]
    command: str
    raw: str


@dataclass(frozen=True)
class Snapshot:
    robot_mode: int
    error_ids: tuple[int, ...]
    q_deg: tuple[float, ...]
    pose_real_mm_deg: tuple[float, ...]
    raw_responses: dict[str, str]


def _normalise_command(command: str) -> str:
    return command.strip().rstrip(";").lower()


def parse_dashboard_response(
    raw: str, *, expected_command: str | None = None
) -> DashboardResponse:
    """Parse one ``ErrorID,{payload},Command(...);`` response safely."""

    text = raw.strip()
    if not text.endswith(";"):
        raise DashboardProtocolError(f"response is missing ';': {raw!r}")
    body = text[:-1]
    try:
        error_text, remainder = body.split(",{", 1)
        payload, command = remainder.rsplit("},", 1)
    except ValueError as exc:
        raise DashboardProtocolError(f"malformed Dashboard response: {raw!r}") from exc

    try:
        error_id = int(error_text.strip())
    except ValueError as exc:
        raise DashboardProtocolError(f"invalid response ErrorID: {raw!r}") from exc

    values_text = payload.strip()
    if not values_text:
        values = ()
    else:
        values = tuple(item.strip() for item in values_text.split(","))
        if any(item == "" for item in values):
            raise DashboardProtocolError(f"empty response value: {raw!r}")

    command = command.strip()
    if not command.endswith(")"):
        raise DashboardProtocolError(f"malformed echoed command: {raw!r}")
    if expected_command is not None and _normalise_command(
        command
    ) != _normalise_command(expected_command):
        raise DashboardProtocolError(
            f"response command {command!r} does not match {expected_command!r}"
        )
    return DashboardResponse(error_id, values, command, text)


class DashboardClient:
    """Small allow-listed Dashboard client used by the capture loop."""

    def __init__(self, sock: SocketLike, *, timeout: float = 2.0):
        self._sock = sock
        self._timeout = timeout
        self._sock.settimeout(timeout)

    @classmethod
    def connect(
        cls, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, timeout: float = 2.0
    ) -> "DashboardClient":
        sock = socket.create_connection((host, port), timeout=timeout)
        return cls(sock, timeout=timeout)

    def close(self) -> None:
        self._sock.close()

    def _receive_response(self) -> str:
        data = bytearray()
        while b";" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Nova closed the Dashboard connection")
            data.extend(chunk)
        response, _remainder = bytes(data).split(b";", 1)
        return response.decode("ascii", errors="replace") + ";"

    def query(self, command: str) -> DashboardResponse:
        if command not in ALLOWED_COMMAND_SET:
            raise PermissionError(f"Dashboard command is not allow-listed: {command!r}")
        self._sock.sendall(command.encode("ascii"))
        return parse_dashboard_response(
            self._receive_response(), expected_command=command
        )


def _parse_float_values(
    response: DashboardResponse, *, count: int, label: str
) -> tuple[float, ...]:
    if response.error_id != 0:
        raise CaptureError(
            f"{label} returned Dashboard ErrorID={response.error_id}; sample refused"
        )
    if len(response.values) != count:
        raise DashboardProtocolError(
            f"{label} returned {len(response.values)} values; expected {count}"
        )
    try:
        values = tuple(float(item) for item in response.values)
    except ValueError as exc:
        raise DashboardProtocolError(f"{label} returned non-numeric values") from exc
    if not all(math.isfinite(value) for value in values):
        raise DashboardProtocolError(f"{label} returned non-finite values")
    return values


def _parse_mode(response: DashboardResponse) -> int:
    values = _parse_float_values(response, count=1, label="RobotMode()")
    mode = values[0]
    if not mode.is_integer():
        raise DashboardProtocolError(f"RobotMode() returned non-integer mode {mode}")
    return int(mode)


def _parse_error_ids(response: DashboardResponse) -> tuple[int, ...]:
    if response.error_id != 0:
        raise CaptureError(
            f"GetErrorID() returned Dashboard ErrorID={response.error_id}; sample refused"
        )
    # GetErrorID uses {[id,...]} as its payload.  parse_dashboard_response
    # leaves the bracket characters on the first/last value for this command.
    if not response.values:
        return ()
    if len(response.values) == 1 and response.values[0] in {"[]", "{}"}:
        return ()
    values = list(response.values)
    if values[0].startswith("["):
        values[0] = values[0][1:]
    if values[-1].endswith("]"):
        values[-1] = values[-1][:-1]
    if not values or any(value == "" for value in values):
        return ()
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise DashboardProtocolError(
            f"GetErrorID() returned malformed error list: {response.raw!r}"
        ) from exc


def read_snapshot(client: DashboardClient) -> Snapshot:
    """Read one ordered, paired robot snapshot using only allowed queries."""

    mode_response = client.query("RobotMode()")
    error_response = client.query("GetErrorID()")
    angle_response = client.query("GetAngle()")
    pose_response = client.query("GetPose(user=0,tool=0)")
    snapshot = Snapshot(
        robot_mode=_parse_mode(mode_response),
        error_ids=_parse_error_ids(error_response),
        q_deg=_parse_float_values(angle_response, count=6, label="GetAngle()"),
        pose_real_mm_deg=_parse_float_values(
            pose_response, count=6, label="GetPose(user=0,tool=0)"
        ),
        raw_responses={
            "RobotMode()": mode_response.raw,
            "GetErrorID()": error_response.raw,
            "GetAngle()": angle_response.raw,
            "GetPose(user=0,tool=0)": pose_response.raw,
        },
    )
    return snapshot


def snapshot_rejection_reason(snapshot: Snapshot) -> str | None:
    """Return why a snapshot must not be saved as a normal sample."""

    mode_name = ROBOT_MODE_NAMES.get(snapshot.robot_mode, "UNKNOWN")
    if snapshot.robot_mode in UNRELIABLE_ROBOT_MODES:
        return (
            f"RobotMode={snapshot.robot_mode} ({mode_name}) indicates initialization, "
            "motion, pause, collision, or alarm; readings are not a stable sample"
        )
    if snapshot.robot_mode not in READABLE_ROBOT_MODES:
        return f"RobotMode={snapshot.robot_mode} ({mode_name}) is unknown"
    if snapshot.error_ids:
        return f"GetErrorID() reports active errors: {list(snapshot.error_ids)}"
    return None


def _new_dataset() -> dict[str, Any]:
    return {
        "metadata": {
            "schema_version": 1,
            "robot": "Dobot Nova2",
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "user": USER,
            "tool": TOOL,
            "pose_orientation_convention": ORIENTATION_CONVENTION,
            "orientation": ORIENTATION_CONVENTION,
            "capture_mode": "read_only",
            "source": SOURCE,
        },
        "samples": [],
    }


def _load_dataset(path: Path, *, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return _new_dataset()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read existing output {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CaptureError("existing output root must be a JSON object")

    # Migrate the original one-sample format in memory so appending never
    # silently discards an earlier observation.
    if "samples" not in document:
        sample = dict(document)
        sample.setdefault("name", "pose_01")
        sample.setdefault("pose_real_mm_deg", sample.get("pose_mm_deg"))
        document = {
            "metadata": {
                "robot": "Dobot Nova2",
                "user": document.get("user", USER),
                "tool": document.get("tool", TOOL),
                "pose_orientation_convention": document.get(
                    "orientation", ORIENTATION_CONVENTION
                ),
                "orientation": document.get("orientation", ORIENTATION_CONVENTION),
                "capture_mode": "read_only",
                "source": SOURCE,
            },
            "samples": [sample],
        }
    if not isinstance(document.get("samples"), list):
        raise CaptureError("existing output 'samples' must be a JSON array")
    metadata = document.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise CaptureError("existing output 'metadata' must be a JSON object")
    metadata.setdefault("user", USER)
    metadata.setdefault("tool", TOOL)
    metadata.setdefault("pose_orientation_convention", ORIENTATION_CONVENTION)
    metadata.setdefault("orientation", ORIENTATION_CONVENTION)
    metadata.setdefault("capture_mode", "read_only")
    metadata.setdefault("source", SOURCE)
    return document


def _next_sample_name(samples: list[Any]) -> str:
    used = {
        item.get("name")
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    index = 1
    while f"pose_{index:02d}" in used:
        index += 1
    return f"pose_{index:02d}"


def _sample_document(name: str, snapshot: Snapshot) -> dict[str, Any]:
    return {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_timestamp_s": time.monotonic(),
        "robot_mode": snapshot.robot_mode,
        "error_ids": list(snapshot.error_ids),
        "q_deg": list(snapshot.q_deg),
        "pose_real_mm_deg": list(snapshot.pose_real_mm_deg),
        "user": USER,
        "tool": TOOL,
        "orientation": ORIENTATION_CONVENTION,
        "source": SOURCE,
        "raw_responses": snapshot.raw_responses,
    }


def _write_dataset(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _print_snapshot(snapshot: Snapshot) -> None:
    mode_name = ROBOT_MODE_NAMES.get(snapshot.robot_mode, "UNKNOWN")
    print(f"RobotMode: {snapshot.robot_mode} ({mode_name})")
    print(f"GetErrorID: {list(snapshot.error_ids)}")
    print(f"GetAngle q_deg: {list(snapshot.q_deg)}")
    print(f"GetPose(user=0,tool=0) mm_deg: {list(snapshot.pose_real_mm_deg)}")
    if snapshot.robot_mode != 5:
        print(
            "WARNING: RobotMode is not 5; this capture is read-only and will not "
            "change the state."
        )


def _capture_loop(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    document = _load_dataset(output, overwrite=args.overwrite)
    try:
        while True:
            try:
                # Do not hold Dashboard port 29999 while waiting for the
                # operator or while the separate teleoperation stack moves
                # the robot.  Other Dashboard clients (including the normal
                # master/slave teleop process) can then use the port.
                client = DashboardClient.connect(timeout=args.timeout)
                print(f"Connected read-only to {DEFAULT_HOST}:{DEFAULT_PORT}.")
                try:
                    snapshot = read_snapshot(client)
                finally:
                    client.close()
                print("Read-only Dashboard connection closed before confirmation.")
                _print_snapshot(snapshot)
            except (
                ConnectionError,
                OSError,
                DashboardProtocolError,
                CaptureError,
            ) as exc:
                print(f"Capture refused: {exc}")
                return 2

            reason = snapshot_rejection_reason(snapshot)
            if reason is not None:
                print(f"SAMPLE NOT SAVED: {reason}.")
            else:
                default_name = _next_sample_name(document["samples"])
                entered_name = input(
                    f"Sample name [{default_name}] (blank uses default): "
                ).strip()
                name = entered_name or default_name
                if not name:
                    print("SAMPLE NOT SAVED: sample name cannot be empty.")
                elif any(
                    isinstance(item, dict) and item.get("name") == name
                    for item in document["samples"]
                ):
                    print(f"SAMPLE NOT SAVED: sample name {name!r} already exists.")
                else:
                    print(f"Ready to append sample {name!r} to {output}")
                    confirmation = input("Type SAVE to confirm this sample: ").strip()
                    if confirmation != "SAVE":
                        print(
                            "SAMPLE NOT SAVED: explicit SAVE confirmation was not given."
                        )
                    else:
                        sample = _sample_document(name, snapshot)
                        document["samples"].append(sample)
                        _write_dataset(output, document)
                        print(f"SAVED sample {name!r} to {output}")
                        print("Saved sample JSON:")
                        print(json.dumps(sample, indent=2))

            again = input("Capture another sample? [Y/n]: ").strip().lower()
            if again in {"n", "no", "q", "quit", "exit"}:
                break
    except KeyboardInterrupt:
        print("\nCapture stopped; no command was sent beyond the read-only allowlist.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON dataset to append samples to"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing output file before capturing",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="TCP read timeout in seconds"
    )
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be a positive finite number")
    try:
        return _capture_loop(args)
    except (ConnectionError, OSError) as exc:
        print(f"Unable to connect to {DEFAULT_HOST}:{DEFAULT_PORT}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
