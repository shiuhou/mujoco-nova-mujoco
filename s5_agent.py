"""Official agent integration; only transport limits/telemetry are added here.

No provider serialization, response parser or motion interpolation is duplicated.
Replay is explicit and still runs the actual official policy and Responses client.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import ssl
import sys
import time
from contextlib import contextmanager, nullcontext
from urllib.parse import urlsplit

import httpx
from inspect_robots_agent.policy import agent_policy

from s5_common import (
    FORBIDDEN_HARDWARE_ADDRESSES,
    Budget,
    S5Error,
    live_settings,
    redact_text,
    register_credentials,
    validate_provider_proxy,
    provider_profile,
    provider_base_url,
    verify_official_stack,
)


class SimulationIOGuard:
    """Python audit backstop, not an OS sandbox. No hardware/device backend."""

    def __init__(self, base_url: str | None = None, *, proxy: str | None = None):
        self.base_url = provider_base_url(base_url)
        self.hostname = urlsplit(self.base_url).hostname
        self.proxy = validate_provider_proxy(self.base_url, proxy)
        self.connect_host = "127.0.0.1" if self.proxy else self.hostname
        self.connect_port = 7890 if self.proxy else 443
        self.active = False
        self.addresses: set[str] = set()
        self.blocked = {"network": 0, "physical_device": 0}

    def check(self, event: str, args: tuple) -> None:
        network_allowed = self.active
        if event == "socket.getaddrinfo":
            network_allowed &= (
                args[0] == self.connect_host and args[1] == self.connect_port
            )
        elif event == "socket.connect":
            address = args[1]
            network_allowed &= (
                isinstance(address, tuple)
                and len(address) >= 2
                and address[0] in self.addresses
                and address[0] not in FORBIDDEN_HARDWARE_ADDRESSES
                and address[1] == self.connect_port
            )
        elif event == "socket.bind":
            network_allowed = False
        if (
            event
            in {"socket.__new__", "socket.getaddrinfo", "socket.connect", "socket.bind"}
            and not network_allowed
        ):
            self.blocked["network"] += 1
            raise S5Error(
                "prohibited_io", "network access outside approved provider transport"
            )
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if any(
                candidate.startswith(
                    (
                        "/dev/video",
                        "/dev/v4l/",
                        "/dev/serial/",
                        "/dev/ttyUSB",
                        "/dev/ttyACM",
                        "/dev/ttyS",
                    )
                )
                for candidate in (path, os.path.realpath(path))
            ):
                self.blocked["physical_device"] += 1
                raise S5Error(
                    "prohibited_io", "physical camera/serial devices prohibited"
                )

    def install(self) -> None:
        sys.addaudithook(self.check)

    @contextmanager
    def provider_network(self):
        self.active = True
        try:
            if self.proxy:
                # Exact operator-approved CONNECT transport boundary. Upstream
                # DNS belongs to the proxy; no fake IP is classified as public.
                self.addresses = {"127.0.0.1"}
                yield
                return
            resolved = socket.getaddrinfo(self.hostname, 443, type=socket.SOCK_STREAM)
            self.addresses = {
                str(result[4][0])
                for result in resolved
                if ipaddress.ip_address(result[4][0]).is_global
                and str(result[4][0]) not in FORBIDDEN_HARDWARE_ADDRESSES
            }
            if not self.addresses:
                raise S5Error(
                    "provider_network_configuration",
                    "no public provider address resolved",
                )
            yield
        finally:
            self.active = False
            self.addresses.clear()


class BoundedTransport(httpx.BaseTransport):
    """Count actual attempts, including official retries; never log raw payloads."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        budget: Budget,
        *,
        live: bool,
        guard: SimulationIOGuard | None = None,
        base_url: str | None = None,
    ):
        self.inner = inner
        self.budget = budget
        self.live = live
        self.guard = guard
        self.base_url = provider_base_url(base_url)
        self.destination = str(httpx.URL(self.base_url + "/responses"))
        if guard is not None and guard.base_url != self.base_url:
            raise S5Error(
                "invalid_provider_configuration", "guard and transport endpoints differ"
            )
        self.events: list[dict] = []
        self.seconds = 0.0
        self.turn_first_request: float | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or str(request.url) != self.destination:
            raise S5Error(
                "provider/API_error", "unexpected provider destination or method"
            )
        self.budget.request()
        started = time.perf_counter()
        if self.turn_first_request is None:
            self.turn_first_request = started
        timeout = min(60.0, self.budget.remaining())
        request.extensions["timeout"] = dict.fromkeys(
            ("connect", "read", "write", "pool"), timeout
        )
        event: dict = {
            "attempt": self.budget.requests,
            "live": self.live,
            "status": None,
            "http_transport_entered": False,
            "request_transmitted": False,
        }

        sending_method = b"POST"

        def trace(name: str, info: dict) -> None:
            nonlocal sending_method
            # httpcore's trace exposes completion boundaries. Never inspect or
            # persist headers. CONNECT completion is not a provider request.
            if name.endswith("send_request_headers.started"):
                sending_method = getattr(info.get("request"), "method", b"")
            if name.endswith("connect_tcp.complete"):
                event["tcp_connected"] = True
            if name.endswith("start_tls.complete"):
                event["tls_completed"] = True
            if name in {
                "http11.send_request_headers.complete",
                "http2.send_request_headers.complete",
            }:
                if sending_method == b"POST":
                    event["request_transmitted"] = True

        request.extensions["trace"] = trace
        try:
            context = (
                self.guard.provider_network()
                if self.guard is not None and self.live
                else nullcontext()
            )
            with context:
                event["http_transport_entered"] = True
                response = self.inner.handle_request(request)
                response.read()
                event["request_transmitted"] = True
            event["status"] = response.status_code
            # Explicit metadata allowlist; no output/reasoning items or headers.
            credential = request.headers.get("authorization", "").removeprefix(
                "Bearer "
            )
            register_credentials({"API_KEY": credential})
            try:
                payload = response.json()
            except ValueError:
                raise S5Error(
                    "provider/API_error", "provider returned invalid JSON"
                ) from None
            if isinstance(payload, dict):
                claimed_model = payload.get("model")
                if isinstance(claimed_model, str) and re.fullmatch(
                    r"[A-Za-z0-9._/-]{1,100}", claimed_model
                ):
                    event["relay_claimed_model"] = redact_text(claimed_model)
                request_id = response.headers.get("x-request-id")
                if (
                    request_id
                    and re.fullmatch(r"req_[A-Za-z0-9_-]{1,100}", request_id)
                    and (not credential or credential not in request_id)
                ):
                    event["request_id"] = request_id
                response_id = payload.get("id")
                if (
                    isinstance(response_id, str)
                    and re.fullmatch(r"resp_[A-Za-z0-9_-]{1,150}", response_id)
                    and (not credential or credential not in response_id)
                ):
                    event["response_id"] = response_id
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    event["usage"] = {
                        key: usage[key]
                        for key in ("input_tokens", "output_tokens", "total_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0
                    }
            if response.status_code != 200:
                # Preserve only ordinary provider error fields, with credential
                # redaction before any persistence or exception construction.
                error = payload.get("error") if isinstance(payload, dict) else None
                safe_error = {}
                if isinstance(error, dict):
                    for key in ("type", "code", "param", "message"):
                        value = error.get(key)
                        if isinstance(value, str):
                            safe_error[key] = redact_text(value)
                if safe_error:
                    event["provider_error"] = safe_error
                raise S5Error(
                    "provider/API_error",
                    f"provider HTTP status {response.status_code}"
                    + (
                        ": " + json.dumps(safe_error, ensure_ascii=False)
                        if safe_error
                        else ""
                    ),
                )
            return response
        except httpx.TimeoutException:
            event["error"] = "model_timeout"
            raise S5Error("model_timeout", "provider request timed out") from None
        except S5Error as exc:
            event["error"] = exc.category
            raise
        except httpx.ConnectError as exc:
            cause: BaseException | None = exc
            tls_failure = False
            while cause is not None:
                tls_failure |= isinstance(cause, ssl.SSLError)
                cause = cause.__cause__
            event["error"] = (
                "TLS"
                if tls_failure
                else "proxy_connect"
                if self.guard and self.guard.proxy
                else "provider_connect"
            )
            raise S5Error(
                event["error"],
                "provider TLS verification failed"
                if tls_failure
                else "approved transport connection failed",
            ) from None
        except httpx.ProxyError:
            event["error"] = "proxy_connect"
            raise S5Error("proxy_connect", "approved proxy CONNECT failed") from None
        except Exception:  # noqa: BLE001 - suppress arbitrary transport/credential text
            event["error"] = "provider/API_error"
            raise S5Error("provider/API_error", "provider transport failed") from None
        finally:
            elapsed = time.perf_counter() - started
            self.seconds += elapsed
            event["T_model"] = elapsed
            self.events.append(event)

    def close(self) -> None:
        self.inner.close()

    @property
    def transmitted_requests(self) -> int:
        return sum(bool(event["request_transmitted"]) for event in self.events)


def make_live_policy(
    embodiment,
    budget: Budget,
    model: str | None,
    env: dict[str, str],
    *,
    effort: str = "medium",
    guard: SimulationIOGuard,
    base_url: str | None = None,
):
    """Official provider with explicit endpoint; no fake or provider fallback."""
    register_credentials(env)
    profile = provider_profile(base_url, env)
    requested = live_settings(model, env, base_url=base_url)
    if not env[profile["api_key_env"]].strip():
        raise S5Error(
            "missing_api_key",
            "selected provider credential must be set by the operator",
        )
    provider_base_url(base_url)
    verify_official_stack()
    proxy = profile["proxy"]
    if guard.proxy != proxy:
        raise S5Error(
            "invalid_provider_configuration",
            "guard and transport proxy settings differ",
        )
    transport = BoundedTransport(
        httpx.HTTPTransport(retries=0, trust_env=False, proxy=proxy, verify=True),
        budget,
        live=True,
        guard=guard,
        base_url=base_url,
    )
    try:
        policy = agent_policy(
            model=requested,
            base_url=base_url,
            wire="responses",
            api_key_env=profile["api_key_env"],
            env={profile["api_key_env"]: env[profile["api_key_env"]]},
            wire_capture=False,
            transcript_echo=False,
            images="always",
            depth="off",
            image_horizon=2,
            effort=effort,
            max_llm_calls=budget.limits.max_llm_calls,
            max_speed_frac=0.1,
            pre_check=embodiment.pre_check,
            transport=transport,
        )
        policy.bind(embodiment.info)
    except Exception:  # noqa: BLE001 - constructor exceptions may echo configuration
        transport.close()
        raise S5Error(
            "provider_configuration_error", "official agent initialization failed"
        ) from None
    return policy, transport


def make_replay_policy(embodiment, budget: Budget):
    """Explicit OFFLINE test fixture through the real official agent implementation."""
    verify_official_stack()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        payload = json.loads(request.content)
        # Parse only the EEF state text that the official policy sent. This fake
        # test response does not have a simulator/object handle or task trajectory.
        states = [
            part["text"]
            for message in payload["input"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "input_text" and "eef_state" in part.get("text", "")
        ]
        match = re.search(r"\bz=([-+0-9.eE]+)", states[-1])
        if match is None:
            raise S5Error("offline_fixture_error", "official EEF label format changed")
        delta = 0.005 if calls == 0 else -0.005
        calls += 1
        return httpx.Response(
            200,
            json={
                "id": f"offline_fixture_{calls}",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"offline_move_{calls}",
                        "name": "move_to",
                        "arguments": json.dumps(
                            {
                                "targets": {
                                    "z": float(match[1]) + delta,
                                    "gripper": 1.0,
                                },
                                "note": "Offline fixture: a bounded EEF-relative vertical move.",
                            }
                        ),
                    }
                ],
            },
        )

    transport = BoundedTransport(httpx.MockTransport(respond), budget, live=False)
    policy = agent_policy(
        model="offline-fixture-not-astra",
        base_url="https://api.openai.com/v1",
        env={},
        wire="responses",
        wire_capture=False,
        transcript_echo=False,
        images="always",
        depth="off",
        image_horizon=2,
        effort="medium",
        max_llm_calls=budget.limits.max_llm_calls,
        max_speed_frac=0.1,
        pre_check=embodiment.pre_check,
        transport=transport,
    )
    policy.bind(embodiment.info)
    return policy, transport


def observable_tools(policy) -> list[dict]:
    """Persist action/tool data only; exclude assistant prose and private reasoning."""
    events = []
    for message in policy.transcript() or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            name = function.get("name")
            if name not in {"move_to", "done", "give_up"}:
                events.append({"tool": "unsupported"})
                continue
            event = {"tool": name}
            try:
                arguments = json.loads(function.get("arguments", "{}"))
                targets = arguments.get("targets", {})
                event["targets"] = {
                    label: value
                    for label, value in targets.items()
                    if label in {"x", "y", "z", "yaw", "pitch", "roll", "gripper"}
                    and type(value) in (float, int)
                    and abs(value) < 1e6
                }
            except (ValueError, TypeError, AttributeError):
                event["invalid_arguments"] = True
            events.append(event)
    return events
