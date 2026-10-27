"""Python audit backstop for S4 CLI/tests; permits offscreen graphics, not capture.

This is additional instrumentation, not an OS security sandbox. EGL may access
graphics driver nodes. Camera/serial device nodes and Python sockets are denied.
"""

from __future__ import annotations

import os
import sys


def install() -> dict[str, int]:
    blocked = {"socket_attempts": 0, "physical_device_attempts": 0}

    def audit(event: str, args: tuple) -> None:
        if event in (
            "socket.__new__",
            "socket.connect",
            "socket.bind",
            "socket.getaddrinfo",
        ):
            blocked["socket_attempts"] += 1
            raise RuntimeError("S4 offline guard prohibits network sockets")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            # Resolve aliases so symlinks cannot turn an innocent path into a device.
            paths = (path, os.path.realpath(path))
            if any(
                p.startswith(
                    (
                        "/dev/video",
                        "/dev/v4l/",
                        "/dev/serial/",
                        "/dev/ttyUSB",
                        "/dev/ttyACM",
                        "/dev/ttyS",
                    )
                )
                for p in paths
            ):
                blocked["physical_device_attempts"] += 1
                raise RuntimeError(
                    "S4 offline guard prohibits physical camera/serial devices"
                )

    sys.addaudithook(audit)
    return blocked
