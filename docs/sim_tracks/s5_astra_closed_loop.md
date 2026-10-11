# S5 — official Astra agent to unified MuJoCo

S5 validates the model-facing closed loop in simulation.
It does NOT authorize autonomous physical Nova execution.

Current live status: **S5A PASS; S5B post-fix live replication PASS; S5 PASS / FROZEN** (October 11, 2026).
Attempt #1 remains PARTIAL and the October 10 proxy-blocked invocation remains
BLOCKED. The separately authorized October 11 resumption completed one live
episode successfully; see the final section and
`evidence/sim_s5b/20261011T040204Z-postfix-live/`.
The S5A live episode used the pinned official agent with configured and
relay-reported `gpt-6-astra`: two Responses requests, two `move_to` decisions,
four validated waypoints and fresh generations 1 → 2 → 3. See the S5A live
PASS record at the end and `evidence/sim_s5a/20261009T124559Z-live/`.
PRE_LIVE_S5_HEAD is `02c80a7c6ea840dfe9d7fc4f236e27330269c1cf`.
The original OFFLINE_S5_HEAD remains
`2689e4a772d6eb8ec48bc5df0aba220944f916d0`. Earlier blocked attempts below are
retained as historical evidence; no live request preceded the offline freeze.
The full benchmark instruction remains:

> Pick up the blue block and place it inside the red box.

## Base checkpoint and scope

Work starts at `2bf4650d08ae3812836c3562defc16976a27db70` on
`research/astra-mujoco-embodiment`. S3/S4 integration evidence at
`evidence/sim_s3_s4/20260927T162753Z/` is PASS. The 176-test integrated baseline
was independently rerun before edits and passed. Frozen simulator, IK, gripper,
benchmark geometry, camera code, and all earlier phase tests are unchanged.
The S3/S4 source checkpoint remains the authority for contact physics and
camera geometry; nothing here establishes physical fidelity.

## Actual official stack and source isolation

The inspected monorepo is `robocurve/references/inspect-robots`, revision
`7e506e3b6f39530f18e75cf97b23c4287aa90f12`. Its bundled agent is version 0.27.0.
The public entry point is `inspect_robots_agent.policy:agent_policy`, producing
`LLMAgentPolicy`. The lifecycle used is `bind(EmbodimentInfo)`, `reset(Scene)`,
then `act(Observation) -> ActionChunk`. The embodiment implements the official
`reset/step/close` interface and adds whole-chunk execution for this loop.

The inspected YAM reference revision is
`178c93037afcf593618e0a1db6ae1f1b0cfeaa0f`; its EEF agent integration test was
used to confirm the public interface. YAM hardware code is not imported.

The historical official checkout has unrelated local edits in `_tools.py` and
`approver.py`. S5 uses a clean `git archive` export of the committed revision,
installed into isolated `.venv-s5`. It neither modifies that checkout nor copies
its dirty edits. `configs/s5_official_stack.json` fingerprints all 55 official
Python source files across both packages. A mismatch fails before import/use.
The core archive's distribution version is `0.0.0` because VCS metadata is
absent; the exact commit and source-tree digests, not that fallback version,
are the provenance authority.

```text
official agent_policy / LLMAgentPolicy
  -> official ResponsesClient (provider resolution, images, response parsing)
  -> generic move_to(targets={dimension: absolute_value}, note=...)
  -> official bounded interpolation -> ActionChunk
  -> S5 whole-chunk validation and isolated IK preflight
  -> ManipulationChunkExecutor, a NovaMujocoChunkExecutor subclass
  -> unchanged S3 contact simulation -> deterministic settle/hold
  -> fresh S4 observation -> next official policy decision
```

There is no new Astra HTTP client, response parser, command language, or model
motion interpolator. `BoundedTransport` is an httpx transport decorator; the
official Responses client still creates and parses the wire messages.
The runner owns the small closed loop because it must validate and execute
whole chunks, settle fully, and prove fresh observations between decisions.
It does not use a high-rate policy rollout.

## Configuration and fail-closed live execution

Accepted configuration was verified from the pinned source:

| Setting | S5A value |
|---|---|
| Factory | `inspect_robots_agent.policy.agent_policy` |
| Provider | OpenAI Responses wire; direct OpenAI or explicit operator-approved HTTPS relay; no fallback |
| Model | explicit bare `--model gpt-6-astra` with `--base-url`; direct routing requires `openai/` prefix |
| Checkpoint target | operator-confirmed `gpt-6-astra`; live account access is tested only by the bounded run |
| Wire | `responses` |
| Effort | `medium` by default, configurable using the official accepted names |
| Images / image horizon | `always` / `2` |
| Depth | `off` |
| Wire capture / transcript echo | both disabled |
| Maximum model requests | 2 by default, CLI maximum 3, counting actual transport attempts |
| Maximum decisions | 2 by default and maximum |
| Maximum episode waypoints | 32 by default, CLI maximum 64 |
| Maximum wall time | 90 s by default, CLI maximum 300 s |

The operator explicitly selected `gpt-6-astra` and the HTTPS relay above.
With an explicit base URL, the pinned official resolver keeps the model string
unchanged and the official `ResponsesClient` posts to `<base>/responses`.
No alias, provider substitution, custom serialization or response parser is
introduced. A provider rejection stops immediately and is reported.

Live preflight checks the integrated ancestor/evidence, credentials, explicit
model, source fingerprints, and current test/lint/type/benchmark results.
There is no implicit `.env` loader: export `OPENAI_API_KEY` manually in the
runner's environment. `.env.example` contains only `OPENAI_API_KEY=`; real
`.env` files are ignored. Do not send the key in a chat, argument or evidence.

The transport accepts only POST to the configured HTTPS base plus `/responses`.
The endpoint may not carry credentials, query strings or fragments. It uses
port 443, public DNS destinations, and an explicit controller-address denylist
because the local Nova address is outside RFC1918. The audit guard pins DNS
and connection attempts to that endpoint during the provider transport.
Unexpected destinations, redirects, robot ports, listeners, physical cameras
and serial devices are denied. Ambient proxies and alternate-provider keys are
not used. EGL graphics access is allowed. This is a Python audit backstop,
not an OS sandbox or physical safety certification.

Every actual request attempt consumes the shared budget before transmission.
Transport errors and non-200 responses stop immediately with sanitized
categories; the official client's retry code cannot create uncounted requests.
The official policy may re-prompt after an invalid tool output, but those calls
share the same maximum. Per-request timeouts are bounded by remaining wall
time; a main-thread Linux signal deadline also interrupts a blocking request,
render or loop. Source/renderer setup is included in the episode wall budget.

## Observation contract and freshness

| Camera | Shape | Representation |
|---|---|---|
| `front_rgb` | `(480,640,3)` | contiguous, read-only uint8 RGB |
| `right_rgb` | `(480,640,3)` | contiguous, read-only uint8 RGB |
| `wrist_rgb` | `(720,1280,3)` | contiguous, read-only uint8 RGB |

Policy state is **only** `eef_state`: base x/y/z metres, reset-relative
yaw/pitch/roll radians, and measured normalized gripper aperture. The adapter
does not falsify the measured aperture or tiny orientation residuals. The
official motion tool's pinned orientation bounds emit exactly zero orientation
actions. `extra` contains only the runtime `env_step` counter.

No `joint_pos`, object poses, object IDs, contacts, segmentation, depth,
`inside_target`, task success, or simulator/camera transforms enter the model
observation, embodiment notes, or scene metadata. Evaluator-only initial object
poses and final metrics are written to local evidence, outside policy inputs.
The S5A instruction uses the observed robot position, not numeric object targets.

The adapter owns a single current observation and permits one decision to
consume it. Execution invalidates it before motion; reset also invalidates it.
An S4 snapshot/hash comparison detects state changes since capture. After
execution/settle, all three images are rendered anew and must share generation,
state hash and time. A failed render removes the reusable observation; no stale
fallback is possible. Generations remain monotonic within the rig lifecycle.
The next decision receives the actual returned post-settle observation. S4
metadata and image hashes prove this in evidence without entering model state.

## Actions, validation and settle

The public seven-value order remains `[x,y,z,yaw,pitch,roll,gripper]`: metres,
radians, normalized aperture; 0 closed / 1 open. There is no Dobot-specific
tool, raw control, qpos or joint access for the model. The nominal grasp center
is flange +Z197 mm with `measured_tcp=false`.

The official agent refuses Euler interpolation. Because all three rotation
slots are pinned to zero, the embodiment accurately advertises
`rotation_repr="none"` with explicit yaw/pitch/roll labels and zero bounds.
It does not advertise `rot6d` or arbitrary orientation support. Both the
official tool and the backend reject nonzero orientation requests.

S5A uses an explicit simulation engineering workspace:
base x `[-.65,.05]`, y `[-.45,.35]`, z `[.015,.45]` metres. These are bounding
limits, not a guarantee every point is reachable. Isolated sequential IK
preflight checks every waypoint before the chunk can change simulation state.
The unchanged S3 executor validates and solves again at execution time.

Additional S5A limits:

- At most 0.020 m Euclidean displacement per waypoint, preserving S2/S3.
- Official interpolation declares 0.005 m per-axis maximum steps; even a
  diagonal waypoint is below 0.020 m.
- At most 0.010 m requested path per decision and 0.020 m per episode.
- Gripper must end fully open at exactly 1. Intermediate official interpolants
  may differ from 1 by at most `1e-8`, accounting for measured numerical aperture
  compliance without rewriting actions. Range validation remains strict [0,1].
- Finite `(N,7)` shape, bounded chunk length, supported orientation, gripper,
  workspace, ordered displacements, full-chunk remaining step budget, and IK
  are checked before execution. Data/order are copied after validation.
- Failed actual waypoints stop the remainder; no reorder, retry, partial
  remainder execution, hidden weld, object teleport or S3 script is introduced.

Each waypoint uses S3 arm settle. After a completed chunk, the simulator holds
for the remaining aperture travel at its existing slew speed plus 0.25 s.
Both use fixed simulation ticks, not host sleeping. The extra hold uses only
robot gripper state and never checks hidden object coordinates to control motion.

S5A's instruction asks the model to compute a 5 mm upward EEF move from its
observation and, after fresh observation, a 5 mm downward move, both open.
The official agent determines targets and generates ActionChunks. Replay uses
an explicit deterministic response fixture for those same two small motions.
Replay is always `OFFLINE_LOOP_PASS` at best, never S5A PASS.

## Evidence and timing

Every runner output directory is exclusive and never overwritten. Required
artifacts are `validation.json`, `live_agent_result.json`,
`observation_trace.jsonl`, `action_trace.jsonl`, `timing.json`, `environment.json`
and `quality_gates.json`; `episode_trace.jsonl` and final native PNGs are also
saved after a completed loop. A blocked preflight writes empty traces and an
explicit BLOCKED result, not invented observations or timings.

Environment evidence includes source HEAD and hashes (important for uncommitted
work), official revisions/tree digests, Python/MuJoCo/dependency versions,
effective policy config, camera/renderer and benchmark versions, budgets, and
evaluator-only initial q/object states in the loop result. It contains no key.
Action evidence includes exact ordered samples, execution counts, failed index,
IK status/iterations/residuals, generation references and timestamps.

Normal tool names/numeric targets and safe request IDs/token counts are retained.
Only redacted provider error `type`, `code`, `param`, and `message` fields may
be saved. No arbitrary raw provider body, request headers, complete wire response, assistant prose,
private reasoning, encrypted reasoning item or hidden chain-of-thought is saved.
The official Responses client may retain encrypted reasoning internally for
protocol replay; S5 disables wire capture and never exports it.

| Timing field | Measured boundary |
|---|---|
| `T_render` | complete S4 observe calls, including initial observation |
| `T_encode_or_request_prep` | act entry to first transport attempt per decision |
| `T_model` | transport, DNS/TLS if needed, response read; mock latency in replay |
| `T_policy_parse` | remaining act time after prep/model/pre-check work; includes official parsing, interpolation and internal re-prompt preparation |
| `T_chunk_validation` | validation/pre-check time excluding measured IK |
| `T_IK` | isolated preflight and execution-time IK solves |
| `T_execute` | executor overhead excluding measured IK and settle |
| `T_settle` | original settle and deterministic post-chunk hold, including physics |
| `T_total_step` | decision through completed fresh observation or failure |

The episode duration also includes initialization and initial rendering; the
sum of step totals excludes those initial costs. These are host timings, not
simulation seconds. Offline mock timings cannot determine whether live Astra
will be model-bound, render-bound, motion-bound or mixed.

S5A retains explicit `invalid_action`, `IK_fail`, `workspace_reject`,
`model_timeout`, `provider/API_error`, `max_calls_reached`, `max_steps_reached`,
and `safety_validation_abort`. Additional categories include stale/failed
observations, model/key/source/quality preflight failures, wall budget and
policy stop. A model `done` is never geometric success. S3's grasp/lift/drop/
place/release/distractor evaluator remains unchanged. S5B perception/approach/
grasp/transport failure attribution is deferred with S5B; no visual diagnosis
or success is fabricated from a short S5A open-gripper run.

## Reproduce

Create a clean pinned official export and isolated environment. Do not install
the dirty reference checkout into S5:

```bash
cd /home/dsa/project/dobot/nova_mujoco
rev=7e506e3b6f39530f18e75cf97b23c4287aa90f12
target=".s5-deps/inspect-robots-$rev"
mkdir -p "$target"
git -C ../robocurve/references/inspect-robots archive "$rev" | tar -x -C "$target"
uv venv --python /home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python .venv-s5
uv pip install --python .venv-s5/bin/python -r requirements-s5.txt
uv pip install --python .venv-s5/bin/python --no-deps \
  -e "$target" -e "$target/plugins/inspect-robots-agent"

# Use new, unique output directories for every invocation.
.venv-s5/bin/python run_s5_quality.py --output /tmp/nova-s5-quality-UNIQUE
MUJOCO_GL=egl .venv-s5/bin/python run_s5a.py --mode replay \
  --quality-gates /tmp/nova-s5-quality-UNIQUE/quality_gates.json \
  --output evidence/sim_s5a/UNIQUE-offline
```

For this live checkpoint, load the supplied credential into the runner process
as `RELAY_API_KEY` without displaying or persisting it, and pass the confirmed
model and endpoint explicitly. The credential source is not committed here.

```bash
# Only after both operator-supplied environment values and current gates exist:
MUJOCO_GL=egl .venv-s5/bin/python run_s5a.py --mode live \
  --model gpt-6-astra --base-url https://sub.scitrace.cc/ \
  --provider-proxy http://127.0.0.1:7890 \
  --provider-env-file /path/to/operator-credential.env \
  --quality-gates /tmp/nova-s5-quality-UNIQUE/quality_gates.json \
  --max-llm-calls 2 --max-episode-steps 32 --max-decisions 2 \
  --max-wall-time 180 --output evidence/sim_s5a/UNIQUE-live
```

An explicit `--model` overrides `INSPECT_ROBOTS_MODEL`. Missing credentials,
missing model or stale quality gates return exit code 2 before provider/network
construction. The live factory accepts no injected fake/replay transport.
Offline tests inject mock transports independently and never access OpenAI.

## Results and remaining boundary

Current evidence paths and final quality counts are recorded in the October 9
live completion record below. The S5A live gate has passed. A subsequent,
separately scoped S5B implementation and attempt must use the exact benchmark
instruction, autonomous official-agent decisions, and S3 geometric/contact
evaluation. It must not feed the scripted S3 trajectory or hidden coordinates
to the model. No S5B evidence or success claim is produced here.

Future real execution remains separately gated by `pre_check`, approver,
one-use authority, bounded MovL, settle and fresh physical observation, with
the physical E-stop as the primary emergency stop. S5 implements none of that
physical execution and removes none of those requirements.

Existing limitations remain: primitive assets, unmeasured TCP, engineering
camera extrinsics/FOV, simplified gripper/contact dynamics and prescribed arm
references. The bounded live route and its latency are now measured. Upstream
model identity beyond the relay claim, visual task competence and autonomous
pick-and-place success remain unvalidated.

The pinned Responses policy does not expose a Responses output-token cap in
its accepted constructor configuration. Request count and wall time are
bounded; they are not a hard dollar or output-token spending limit.

## Historical pre-credential record — September 28, 2026

- **S5A live: BLOCKED.** The credential is absent and no explicit Astra model
  identifier is configured. No live request occurred.
- **S5A offline implementation: validated.** Explicit replay ran the actual
  pinned official agent and Responses client, generic `move_to`, interpolation,
  whole-chunk validation, inherited executor, physics, settle and fresh feedback.
- **S5B: BLOCKED / not started.** No autonomous pick-and-place episode occurred.
- **224 tests passed in 40.41 s:** all 176 existing tests plus 48 new S5 tests.
  Ruff lint/format (41 files), S5 mypy (six runtime/runner modules), compilation
  and diff checks passed. S1/S2 benchmark remains **48/48**, with position
  residual RMS 0.000197614 mm and maximum 0.000534850 mm.

Saved evidence:

- [`20260928T050009Z-offline`](../../evidence/sim_s5a/20260928T050009Z-offline/):
  `OFFLINE_LOOP_PASS`, never a live S5A PASS. Two mock requests through the
  official Responses path, two decisions, four executed waypoints, 0.010 m
  total completed-waypoint path. All loop checks pass. Frame generations are
  1, 2, 3 at simulated times 0, 0.38, 0.76 s; each hash changes, all three camera
  images change after each move, and the second decision uses generation 2.
  Passing quality logs and a copy of the S1/S2 benchmark evidence are included.
- [`20260928T050009Z-live-preflight`](../../evidence/sim_s5a/20260928T050009Z-live-preflight/):
  `BLOCKED`, primary reason `missing_api_key`; environment metadata also records
  `model_identifier_configured=false`. Source-matched quality gates are PASS.
  There are no model calls, motion, observation trace entries or invented timing.

The offline run's aggregate host timings are:

| Boundary | Milliseconds |
|---|---:|
| Render, three complete observations | 8.232 |
| Encode / first request preparation | 32.076 |
| Mock transport (not real Astra inference) | 0.667 |
| Official policy parse/interpolation residual | 0.585 |
| Chunk validation, excluding IK | 0.972 |
| IK | 3.863 |
| Executor overhead, excluding IK/settle | 0.755 |
| Settle / physics hold | 58.921 |
| Two decision-to-fresh-observation steps | 105.718 |

The final RGB evidence was visually inspected. No secret or raw wire/reasoning
material is present in the JSON/JSONL evidence. Base and ending repository HEAD
remain `2bf4650d08ae3812836c3562defc16976a27db70`; the implementation, tests,
documentation and evidence are local uncommitted changes with exact source
hashes recorded. No commit or push was performed. Existing root-document and
official-reference dirty work was left untouched.

For this work and both evidence runs:

```text
robot_connections = 0
robot_commands = 0
serial_connections = 0
physical_camera_connections = 0
openai_requests = 0
```

## Offline rollback freeze before live requests

The operator authorized a local rollback checkpoint before paid execution. The unchanged implementation was revalidated with the complete pytest suite, 48/48 S1/S2 benchmark, Ruff lint/format, S5 mypy, compilation/imports and Git diff review. Fresh offline replay evidence is `evidence/sim_s5a/20260928T052717Z-offline-freeze`; the result is `OFFLINE_LOOP_PASS`. No live request preceded this freeze. Source fingerprints match the earlier 224-test checkpoint. The operator has now confirmed the relay endpoint and exact `gpt-6-astra` identifier; that subsequent live configuration is separate from this offline rollback snapshot.

The subsequent endpoint configuration adds offline tests for exact model forwarding,
no endpoint fallback, credential-free provider errors, and denied hardware/private
destinations. The official source pin and all embodiment validation stay unchanged.

## Current S5A blocked checkpoint — September 28, 2026

OFFLINE_S5_HEAD is `2689e4a772d6eb8ec48bc5df0aba220944f916d0`.
The reviewed endpoint configuration and its offline replay were committed as
`863994647414ccab52a4cf9343afc7c1eaab0616` before the live-mode attempt.
The branch was clean at both boundaries. No push was performed.

The attempted configuration was `model=gpt-6-astra`, `wire=responses`,
`base_url=https://sub.scitrace.cc/`, `effort=medium`, `images=always`,
`image_horizon=2`, `depth=off`. Limits were two model attempts, two decisions,
32 waypoints and 180 seconds. Gripper remained open; the instruction was the
small up/down S5A check, not the manipulation mission.

At `20260928T053452Z`, the installed official agent processed the integrated
three-camera/EEF observation, prepared its first Responses request, and entered
the bounded transport. DNS returned `198.18.0.65`. The public-destination guard
rejected it with the exact error **`no public provider address resolved`**
before calling the inner HTTP transport. There was no provider response,
authentication result, model-access result or paid inference. Ambient local
proxies were present but were not used; the guard was not relaxed and no
retry, alternate endpoint, different model or replay substitution occurred.

**S5A = BLOCKED. S5B = BLOCKED, not started.** There is no S5A PASS commit.
The model-facing live loop remains unvalidated. The failure is provider network
configuration, not evidence of an unavailable model or failed manipulation.
Correct network routing must be reviewed before resuming bounded S5A. S5B
continues to require a genuine live S5A PASS.

Evidence: [`20260928T053452Z-live`](../../evidence/sim_s5a/20260928T053452Z-live/).
The original runner classified the outcome PARTIAL and counted one transport
attempt as one `openai_request`. Those original artifacts are retained.
[`checkpoint_review.json`](../../evidence/sim_s5a/20260928T053452Z-live/checkpoint_review.json)
provides the explicit correction: **one attempt, zero transmitted requests**.
The runner now records HTTP-transport entry and transmission separately,
using httpcore send-header completion without inspecting trace payloads.
A transmitted request that later times out remains counted.

Only generation 1 exists. No `move_to`, ActionChunk execution, motion, settle,
or post-motion generation occurred. The `after_*` PNGs are the initial/failure
observation because execution never began; `failure_montage.png` labels that
state explicitly. No object coordinates or evaluator data entered the policy.
The original and reviewed evidence contain no credentials or private reasoning.

Measured timings from this attempt (seconds): render `0.00553125`, request
preparation `0.01577333`, DNS/transport gate `0.00164613`, policy residual
`0.00000878`, decision-to-stop `0.01756760`. Chunk validation, IK, execution
and settle were zero. The transport-gate time is **not model inference latency**;
this blocked run cannot establish whether a live loop is model-bound.

Physical robot connections, robot commands, serial connections, physical camera
connections and transmitted OpenAI requests all remain **zero**. S6 is not
started. Future physical execution still requires the separate safety gates
already described; this result grants no physical authority.

### Postflight regression and freeze

After correcting request-count telemetry, **241 tests passed in 40.83 s**:
the original 176 tests, the original 48 S5 tests and 17 endpoint/telemetry
regressions. S1/S2 remains 48/48. S3 manipulation, S4 observations and integrated
S3/S4 regressions pass in the full suite. Ruff lint/format, S5 mypy, compilation,
imports, official-source fingerprints and Git diff checks all pass.

The exact-source quality record is saved as `postflight_quality_gates.json`
inside the blocked evidence directory; the original preflight record remains
unchanged. A new postflight replay at
`evidence/sim_s5a/20260928-blocked-postflight-offline/` is `OFFLINE_LOOP_PASS`,
with two mock requests, two decisions, four waypoints and generations 1 → 2 → 3.
It is not a live retry or live PASS. Native failure images were visually reviewed.
No S1–S4 source or original test changed, and unrelated root/reference work
remains untouched.

## Explicit relay proxy and redaction freeze — October 9, 2026

Resumed from `b92fb96`. The literal credential parser is unchanged. It accepts
plain or `export` assignments as data and never sources the operator file.
The two earlier `20260928-shared-provider-live*` directories are preserved:
both stopped at missing credentials with zero transmitted requests.
`20260928-parser-offline` is the earlier successful offline replay.

Direct mode still resolves the configured HTTPS provider and permits only
public addresses on port 443, excluding the physical controller. Special-use
`198.18.0.0/15` remains rejected. Explicit proxy mode is opt-in and permits
only `https://sub.scitrace.cc` through `http://127.0.0.1:7890`. During that
transport scope, the socket guard permits only that loopback host/port;
upstream provider DNS belongs to the authorized proxy. HTTPX retains TLS
verification for the relay, ignores ambient proxies, and uses zero transport
retries. The exact POST `/responses` destination is checked before transmission;
redirect responses stop the operation. Proxy CONNECT headers do not count as
transmitted provider requests. TCP and TLS completion flags are safe telemetry.
The Python audit guard is a backstop, not an operating-system sandbox.

Shared real/sim code remains the pinned `inspect_robots_agent.policy.agent_policy`
factory, `LLMAgentPolicy`, provider resolution and `_responses.ResponsesClient`,
plus the official generic `move_to` tools and ActionChunk interpolation.
S5 adds guarded HTTPX transport, budgets and metadata only. It imports no real
motion backend. Configured model is explicitly `gpt-6-astra`, wire `responses`,
relay credential variable `RELAY_API_KEY`, effort `medium`, images `always`,
image horizon 2 and depth `off`. A relay model claim is recorded separately
from the requested model; it cannot independently prove upstream model identity.

Credentials from the process environment and file-loaded runtime environment
enter an in-memory redaction set. Exceptions, normal log records, provider
error fields and recursively sanitized JSON/JSONL (including command metadata)
use that set. Raw log exception chains are suppressed; wire capture and
transcript echo stay disabled. No secret or credential file is committed.

Pre-live validation: **275 pytest tests passed (99 S5), S1/S2 48/48**, Ruff,
format, S5 mypy, compilation and diff checks PASS. This includes all frozen
S1-S4 tests. A content audit of 310 tracked/untracked repository artifacts
found zero credential matches before adding this documentation. Fresh replay
`evidence/sim_s5a/20261009-prelive-offline` is `OFFLINE_LOOP_PASS`, with two
mock requests, two decisions and fresh generations 1 → 2 → 3. Its quality
record binds the exact implementation bytes used for the subsequent live run.
This replay makes no live success claim.

The authorized live checkpoint is one S5A episode with at most two requests,
two decisions, 32 waypoints and 180 seconds. S5B is not started by this work.
S5 validates the model-facing closed loop in simulation. It does NOT authorize
autonomous physical Nova execution.


## S5A live PASS — October 9, 2026

This result supersedes the earlier blocked-live status. Starting implementation
HEAD was `b92fb96`; PRE_LIVE_S5_HEAD is
`02c80a7c6ea840dfe9d7fc4f236e27330269c1cf`. The repository was clean before
transmitting credentials. Exactly one live episode ran, with no retry or
provider/model/proxy substitution. Evidence is
[`20261009T124559Z-live`](../../evidence/sim_s5a/20261009T124559Z-live/).
The local commit containing this section and evidence is the S5A checkpoint;
its SHA is reported in the completion message rather than embedded recursively.

The configured provider was SciTrace `https://sub.scitrace.cc/`, explicit proxy
`http://127.0.0.1:7890`, wire `responses`, configured model `gpt-6-astra`.
Both provider responses also reported `gpt-6-astra`. This is relay-reported
identity; S5 cannot independently attest which upstream weights the relay used.
The pinned official revision remained
`7e506e3b6f39530f18e75cf97b23c4287aa90f12`. Effort was medium, images always,
image horizon 2, depth off, transcript echo and wire capture off. Credentials
were loaded from the supplied operator file as data into `runtime_env` and
registered for redaction; only their presence was recorded.

Both requests were transmitted and returned HTTP 200. TCP and verified TLS
completion were observed on the first request; the second reused the connection.
The official policy emitted two `move_to` calls, first raising the observed EEF
by 5 mm and then returning it by 5 mm, with gripper 1 and zero relative
orientation. Its official interpolation created two ActionChunks of two ordered
waypoints each. Whole-chunk validation and isolated IK preflight passed. All
four execution waypoints converged and completed through
`ManipulationChunkExecutor(NovaMujocoChunkExecutor)`. Total Cartesian path was
approximately 10 mm. The gripper stayed open.

Deterministic settle completed after both decisions; simulation times were
0.00 → 0.38 → 0.76 seconds. Fresh generations were **1 → 2 → 3**. State hashes
changed from `14124e228981…` to `0abbf628b3a4…` to `32e083bd2d0d…`; complete
hashes and per-camera hashes are in `observation_trace.jsonl`. All three RGB
frames changed after each action. Decision 2 consumed generation 2. The
model-facing observations remained exactly front/right RGB at 480×640×3,
wrist RGB at 720×1280×3, HWC uint8 RGB, plus seven-value `eef_state`; evaluator
object poses, contacts and success data were not policy inputs.

Limits of two requests, two decisions, 32 waypoints and 180 seconds were all
respected: actual counts were **2 requests, 2 decisions, 4 waypoints**, duration
**24.037959 s**. The safe usage totals were 16,496 input tokens, 216 output tokens,
16,712 total tokens. Response IDs and individual usage are recorded; no private
reasoning or raw responses are persisted.

| Measured component | Seconds |
| --- | ---: |
| Render, all observations | 0.012403 |
| Request preparation | 0.031296 |
| Provider/model path | 23.625336 |
| Policy parse/interpolation residual | 0.000882 |
| Chunk validation | 0.001046 |
| IK | 0.003616 |
| Execution overhead | 0.000669 |
| Settle / physics hold | 0.054999 |
| Decision-to-fresh-observation steps | 23.730093 |
| Episode including initialization | 24.037959 |

This small run is provider/model-bound. Provider time includes relay/network
latency and model inference, which are not independently measured. Policy parse
is a residual from the public act() boundary. Timings are wall-clock durations;
settle simulation time is separately recorded. The final montage uses the actual
saved generation-3 RGB frames and was visually reviewed.

All S5A exit checks passed. The full postflight regression record accompanies
this evidence. Credential-content audits cover repository/evidence bytes and
the staged commit before recording the checkpoint; zero secret matches were
found. `robot_connections=0`, `robot_commands=0`, `serial_connections=0`,
`physical_camera_connections=0`, `transmitted_provider_requests=2`; there were
zero blocked hardware or network attempts. No push was performed.

**S5B readiness: READY for a separately scoped benchmark implementation and
attempt; S5B has not started and this run does not prove manipulation success.**
The current S5A action limits deliberately require the open gripper and short
moves. Camera geometry and nominal grasp-center calibration remain engineering
assumptions; supported orientation stays the validated reset-relative subset.
This result validates the simulator model-facing loop only. Real deployment
still requires independent pre_check, human approver, one-use authority,
bounded MovL, settle, fresh real observations and physical E-stop as primary
emergency stop. No physical execution or S6 work is authorized by this result.

## S5B implementation and first-attempt protocol — October 9, 2026

The operator authorized proceeding after committed S5A PASS `5fb7d38`.
`run_s5b.py` calls the same runner and official provider factory as S5A; no
provider, parser or interpolation code is duplicated. The simulator evaluator
and RGB montage recorder are in `s5_benchmark.py`. Neither generates actions.
The instruction is exactly **Pick up the blue block and place it inside the red
box.** No scripted S3 trajectory or object coordinates enter policy context.

S5B permits normalized gripper commands [0,1], a 150 mm total Cartesian path per
chunk, 1.8 m per episode and 32 waypoints per chunk. Workspace and 20 mm per
waypoint bound, zero relative orientation, whole-chunk validation, isolated IK,
ordered executor, collision aborts and deterministic settle are preserved.
Official interpolation continues to use 5 mm per Cartesian component. These
mission bounds are declared to the policy without target/object information.
S5A retains its original open-only and short-path defaults.

The first live attempt is capped at 12 transmitted/request attempts (including
internal policy re-prompts), 12 decisions, 384 executed waypoints and 600 seconds.
There is one episode and no automatic episode retry. The relay, proxy, key
source, model, wire, effort and image configuration remain the S5A configuration.
Live S5A evidence and its ancestor checkpoint are required before S5B starts.

Success is defined by the unchanged S3 evaluator: stable contact grasp, lift,
acceptable slip, object inside target and settled, released gripper, acceptable
distractor displacement and no simulator fault; S5 also requires no loop abort
or active holding. A model `done` is never evidence of success. Evaluation stays
outside policy observations, with final object poses in evaluator artifacts only.
Transport/drop stage observations are sampled after each settled chunk and may
miss transient events; S3 max-lift/slip/contact diagnostics run during physics.

Full preflight: 296 tests passed (176 S1-S4, 120 S5), S1/S2 48/48, Ruff,
format, S5 mypy, compilation and diff checks PASS. Offline runner evidence is
`evidence/sim_s5b/20261009-offline-loop`: the two mock vertical moves exercise
fresh generations 1 → 2 → 3 and correctly finish PARTIAL/max_calls_reached
without claiming benchmark success. This fixture contains no manipulation
trajectory and made zero provider requests.


## S5B first autonomous attempt — PARTIAL, October 9, 2026

Live source checkpoint: `1b140d6a748a504b0dfa46c4691a2e74cad9457e`.
Evidence: [`20261009T131911Z-live`](../../evidence/sim_s5b/20261009T131911Z-live/).
There was exactly one live benchmark episode. Twelve transmitted Responses
requests returned HTTP 200; twelve policy decisions produced 236 ordered
waypoints over a 1.069247 m path. Configured and relay-claimed model was
`gpt-6-astra`, using the same official pin/provider/proxy as S5A. Usage totals:
114,556 input tokens, 788 output tokens, 115,344 total. No episode retry,
provider change, scripted trajectory or privileged task input was used.

The blue block was contacted, stably grasped, lifted and transported above the
red box. Maximum lift was **0.142930 m**, maximum slip **0.004507 m**. The final
model decision commanded gripper 1. At the final live observation, however,
measured opening was **0.059613 m** and finger contacts persisted. The unchanged
S3 evaluator reported **released=false, inside_target=false, settled=false,
holding_now=false, success=false**. `holding_now=false` alone is not release:
the S3 detector leaves HOLDING when an opening command is active. Black maximum
displacement was **0.0000170434 m** (0.017 mm), acceptable. No simulator fault
occurred. Primary stop category: **max_calls_reached**. Task-stage symptoms:
**release_fail / place_miss**. All limits were respected. The conservative
automated task classifier reported place_miss because the block was not inside;
the diagnosis below identifies the release timing cause.

All 12 chunks passed whole-chunk validation and isolated IK, executed through
NovaMujocoChunkExecutor, completed the then-configured hold and produced fresh
observations. Generations **1 → 2 → … → 13** and complete state/image hashes are
saved; every next decision used its preceding post-action observation. The
contact and release result exposed that the hold was insufficient to finish
actuator slewing, despite fresh images. No frame was reused after motion.

| Timing component | Seconds |
| --- | ---: |
| Render | 0.047915 |
| Request preparation | 0.174804 |
| Provider/model path | 165.129231 |
| Policy parse residual | 0.006744 |
| Chunk validation | 0.015391 |
| IK | 0.165192 |
| Execute overhead | 0.034602 |
| Settle/physics hold | 3.378482 |
| Decision-to-observation total | 169.199489 |
| Episode total | 169.543063 |

The run is provider/model-bound (about 97% of episode wall time); provider time
includes relay/network and cannot isolate upstream inference. Per-decision
component timings are in `episode_trace.jsonl`. Montage I/O is included in
step totals but not render time. Initial, all twelve post-decision, contact,
lifted, transport and final montages preserve the visual episode.

### Confirmed defects and offline diagnosis

On lift decisions 3 and 6 the model omitted `gripper`. The pinned official
`_tools.Toolset._move_absolute` starts a target from the measured EEF state,
so omitted gripper reused the contact aperture (~0.628), rather than the prior
zero command. Those lifts did not establish a stable grasp. On decision 9,
Astra explicitly included `gripper=0`; stable lift and transport then succeeded.
This is observable official action semantics, not a provider failure. Future
policy-facing robot documentation should explicitly explain omission behavior;
no state falsification or hidden command substitution is appropriate.

The terminal release failed because S5's post-chunk wait estimated gripper
travel from commanded versus measured aperture only. Contact can keep measured
opening near 60 mm while the internal actuator setpoint has slewed closed.
The open command must traverse that internal distance before measured fingers
start moving. The live final command was 95 mm, internal setpoint 50.64 mm and
measured aperture 59.61 mm.

An **offline diagnostic** replayed only the recorded autonomous ActionChunks
through the same simulator boundary and matched every live state hash. It then
held the simulator an additional **1.239333 s**, with no new action/model call.
The actuator reached 95 mm, fingers opened, and the unchanged S3 evaluator
reported released, settled and inside_target true, success true. This is
recorded in `offline_release_diagnostic.json` and the explicitly OFFLINE-labeled
montage. **It does not change the live result to PASS.** No object teleport,
weld, target reset or hidden motion was involved.

The implementation now computes the wait from the larger of commanded-to-
measured and commanded-to-setpoint travel, divided by existing slew speed,
plus the original 0.25 s hold. This uses robot actuator state only. The added
contact-release regression verifies the final observation is captured after
setpoint travel and physical finger opening. It does not weaken action/IK or
collision checks. This fix has not been tested with another live paid episode.
The original live evidence and its source fingerprints remain unchanged.

### Postflight evidence and next research checkpoint

The first postflight suite had one S4 static image identity failure (front_rgb);
295 other tests passed. Reset identity and the other camera repeats passed.
The isolated unchanged S4 test passed immediately afterward. The original
failed quality record and S4 diagnostic flags are retained; the transient GPU
rendering mismatch's root cause is unconfirmed. It is not silently discarded.
Final regression after the settle fix: **297 tests passed (121 S5), S1/S2
48/48, Ruff, format, S5 mypy, compilation and diff checks PASS**. The final
record is saved separately alongside the original exact-source preflight
record. S1-S4 sources/tests were not changed. The passing rerun does not
explain the earlier transient static-image mismatch.

**S5A remains PASS. S5B remains PARTIAL.** Before another paid mission, review
these two action/settle findings and the intermittent S4 static-render check.
No S6 work has started. The proposed next research stage is:

**S6 — Sim/Real Observation & Action Contract Bring-up**

- **S6A:** real observation-only parity for front_rgb/right_rgb/wrist_rgb and
  real eef_state; no model-issued physical motion.
- **S6B:** real Astra shadow mode; generic move_to proposals evaluated by the
  safety stack as WOULD_SEND only, without physical execution.
- **S6C:** supervised one-action physical execution with human review,
  pre_check/approver, one-use authority, bounded MovL, settle and fresh
  observation; physical E-stop remains primary.
- **S6D:** incremental supervised manipulation after the preceding gates.

This is a proposal only. Physical robot, serial and camera counters remain
zero. This S5B episode made 12 provider requests; offline diagnosis made zero.
No changes were pushed. Simulator results do not authorize physical autonomy.


## Controlled post-fix replication preflight — BLOCKED, October 9, 2026

Starting HEAD: `4d199b10bca2a3e1cccb9e6c6f2997c97a1aeba2`, clean branch
`research/astra-mujoco-embodiment`. The operator authorized exactly one new live
S5B episode with the same task, provider/model, cameras, action semantics,
initial scene, evaluator, effort, image horizon and 12-call/12-decision budget.
Only the committed gripper settle correction was to vary from attempt #1.
No production code, policy prompt, renderer or validation rule was changed.

The contact-release settle regression passed. The official source fingerprints
and credential-content audit passed. Before paid execution, ten isolated S4
runs were planned, with an explicit stop on any recurrence. Repetitions 1 and 2
passed. **Repetition 3 failed** in
`test_s4_evidence.py::test_evidence_records_frames_and_side_effects_without_overwrite`:
`reset_repeat_exact.front_rgb=false`. Static repeats passed for all three
cameras; reset repeats for right_rgb and wrist_rgb also passed. This differs
from the earlier static-front mismatch but reproduces the broader front-camera
byte-identity reproducibility issue. Root cause remains unconfirmed. Renderer:
MuJoCo 2.3.7, EGL, NVIDIA GeForce RTX 4090, OpenGL 4.6.0 NVIDIA 595.84.

The operator's stop condition was enforced immediately: **no live episode,
provider request or attempt #3; attempt #2 NOT_RUN**. Remaining isolated runs,
new offline replay and full quality suite were not run after this blocker.
Prior 297-test and 48/48 results remain historical and are not presented as
passing this preflight. No renderer workaround or tolerance change was made.

Evidence: `evidence/sim_s5b/20261009T151042Z-postfix-replication/`, containing
all three isolated S4 reports/images, pytest outputs, source/historical hashes,
credential audit, blocked preflight result and `attempt_comparison.json`.
Attempt #1 files were compared against their preflight hashes and remain
byte-for-byte unchanged. No live observation/action traces or mission montages
were fabricated for this blocked preflight.

| Measure | Attempt #1 | Attempt #2 |
| --- | --- | --- |
| Settle implementation | Old defect | Fixed, not live exercised |
| Call budget | 12 | 12 planned |
| Calls / decisions / waypoints | 12 / 12 / 236 | 0 / 0 / 0 |
| Grasp / transport | PASS / PASS | Not evaluated |
| Maximum lift | 142.9 mm | Not evaluated |
| Release / inside target / success | false / false / false | Not evaluated |
| Duration / provider time | 169.54 s / 165.13 s | Not applicable |
| Tokens | 115,344 | 0 |
| Distractor displacement | 0.017 mm | Not evaluated |
| Result | PARTIAL | NOT_RUN; reproducibility blocker |

S5 remains **S5A PASS / S5B PARTIAL**, with the controlled replication blocked.
No conclusion about live sufficiency of the settle fix can be drawn from this
preflight. S5 is frozen with this blocker preserved; resolving renderer
reproducibility requires separate focused work before another live episode.
No S6, E0 or interface experiments started. Robot connections, robot commands,
serial connections, physical-camera connections and transmitted provider
requests are all zero for this preflight. Commits are local only; no push.

## S5B post-fix replication #2 — BLOCKED, October 10, 2026

After the S4R criterion passed its 1,200-cycle stress validation and the local
quality gates were green, exactly one post-fix S5B invocation was started with
the frozen attempt configuration: `gpt-6-astra`, the pinned official
`inspect-robots-agent`, SciTrace relay `https://sub.scitrace.cc/`, explicit
proxy `http://127.0.0.1:7890`, the operator credential source, the unchanged
benchmark scene and instruction, and 12 model-call / 12-decision limits.

The runner initialized the MuJoCo benchmark and captured the initial
observation, then stopped on the first provider attempt with the precise
failure category **`proxy_connect`** (`approved transport connection failed`).
No HTTP request was transmitted (`transmitted_requests=0`), no model decision
or generic action was produced, and no ActionChunk, validation, IK, executor,
settle or simulator motion occurred. The run used one internal provider attempt
and zero paid requests; its wall time was 5.366 s. The failure is transport
connectivity, not authentication, model access, Responses parsing, policy
output, validation, IK, execution or freshness.

Evidence is `evidence/sim_s5b/20261010T060206Z-postfix-replication/`.
Attempt #1 remains unchanged and PARTIAL. Per the controlled-replication gate,
this exact attempt is preserved and no retry or attempt #3 was run. Therefore
the gripper-settle correction is not live-validated by attempt #2, and S5
remains **S5A PASS / S5B PARTIAL with the post-fix replication blocked**.


## S4R render determinism investigation — resolved, October 10, 2026

The intermittent S4 front-camera identity failure was investigated locally before
any further provider request. A 1,200-cycle stress run covered 300 cycles each
of repeated render, reset plus render, fresh renderer plus render, and motion/debug
then reset plus render. Every record includes simulator state hashes, qpos/qvel,
simulation time, frame generation, camera transforms, image hashes, renderer
metadata and diagnostics for mismatches.

The mismatch is renderer-level, not simulator/reset or camera state: the recurring
front-camera residual was four pixels at one 8-bit intensity level; the largest
history/reset residual was five pixels at one level. Maximum whole-frame mean
absolute difference was 6.510416666666667e-06. State hashes, qpos/qvel, derived
MuJoCo transforms, camera poses and prepared scene inputs were equal. Repeated
readback of the same framebuffer was exact in 1,806 checks. Explicit GPU finish,
dithering changes and frozen scene inputs did not remove the residual; disabling
MSAA removed it in 600 experimental cycles. Production MSAA remains unchanged.

Bit-exact RGB equality is therefore not a valid requirement on this EGL/GPU
stack. The production criterion now requires exact state hash and state values,
exact simulation time, coherent camera metadata, unchanged camera poses, a fresh
advancing generation, and RGB differences no greater than eight pixels at one
8-bit level per view. The raw RGB arrays are never modified, blurred or
quantized. Larger differences, stale generations, changed state, camera swaps,
metadata mismatches and malformed frames fail closed. Focused integrity tests
cover these cases.

Stress evidence: `evidence/sim_s4r/20261009T185711Z/`. The replacement
criterion passed all 1,200 post-change cycles with zero criterion failures.
No physical devices or provider were accessed.

## Resumed S5B post-fix live replication — PASS, October 11, 2026

Source checkpoint: `d22a39a7accba01e0557a5ee5a56cbb5f23f6503`.
Evidence: [`20261011T040204Z-postfix-live`](../../evidence/sim_s5b/20261011T040204Z-postfix-live/).
This was one newly authorized resumption after the October 10 invocation failed
before transmission, not an automatic retry. The earlier PARTIAL and BLOCKED
records remain byte-for-byte unchanged. No production source, S4R criterion,
scene, camera, prompt, action contract, IK, executor, workspace, evaluator or
gripper-settle setting was changed during this resumption.

### Credential-free connectivity diagnosis

The explicit HTTP proxy at `127.0.0.1:7890` was listening. One raw HTTP CONNECT
to `sub.scitrace.cc:443` returned 200, followed by a successful hostname and
default-CA verified TLS 1.3 handshake (5.599 s combined). A second independent
connection through the same `httpx.HTTPTransport(retries=0, trust_env=False,
proxy=approved_proxy, verify=True)` and existing `SimulationIOGuard` completed
CONNECT, verified TLS and an unauthenticated `HEAD /` with HTTP 200 (3.367 s).
Neither probe loaded a credential or sent a Responses/model request. No
redirect was followed and no guard rejection occurred.

The prior 5.002 s failure had TCP connected but no completed TLS handshake or
provider transmission. It is **not reproducible in these two connections**;
the historical trace cannot distinguish a transient proxy/upstream-connect
failure from another pre-TLS transport failure. The exact historical root cause
remains unknown. No proxy, provider, TLS, timeout or network-guard workaround
was needed or applied. `connectivity_preflight.json` preserves this distinction.

### Evaluator result and unchanged configuration

The exact instruction was **Pick up the blue block and place it inside the red
box.** The pinned official agent revision remained
`7e506e3b6f39530f18e75cf97b23c4287aa90f12`, with Responses, configured and
relay-reported `gpt-6-astra`, SciTrace relay, the explicit HTTP proxy,
`RELAY_API_KEY`, effort medium, images always, image horizon 2 and depth off.
All recorded policy, scene, camera, renderer, action-limit and budget settings
match attempt #1. The ceilings remained 12 requests, 12 decisions, 384 total
waypoints, 32 per chunk and 600 seconds.

Ten transmitted requests returned HTTP 200 and produced ten generic `move_to`
decisions and 188 ordered waypoints over a 0.855000 m path. Every chunk passed
validation and isolated IK, executed through NovaMujocoChunkExecutor, settled,
and produced a fresh observation. Generations advanced from 1 through 11.
No privileged task state was supplied to the policy. Evaluator success stopped
the loop with two model calls still available; no model declaration of success
was required.

| Evaluator measure | Result |
| --- | --- |
| Correct object / stable grasp | Blue contact observed / true |
| Lift / maximum lift | true / 0.143755 m |
| Maximum slip | 0.005819 m |
| Transport observed | true |
| Released / active holding | true / false |
| Inside target / settled | true / true |
| Final settled duration | 0.710 s |
| Final gripper aperture | 0.095000 m |
| Maximum distractor displacement | 0.0000170434 m (0.017 mm), acceptable |
| Simulator fault / loop abort | false / false |
| Final evaluator success | true |

The successful final placement does not erase intermediate limitations. Decision
3 omitted gripper, so official `move_to` reused measured aperture (0.6284) and
did not establish a stable lift. Astra then reopened, regrasped and explicitly
kept gripper 0 during lift and transport. The recorder also retains
`dropped_observed=true`: at decision 9, during lowering above the box, the
post-chunk HOLDING flag became false while gripper was commanded closed and
inside_target was still false. This is the recorder's coarse drop heuristic,
not a reconstructed contact trajectory. Decision 10 opened the gripper, and
the final evaluator confirmed release, settled containment and success. Neither
the flag nor the evaluator was modified to obtain PASS.

### Timing, usage and final gates

| Timing component | Seconds |
| --- | ---: |
| Render | 0.041056 |
| Request preparation | 0.145220 |
| Provider/model path | 132.826602 |
| Policy parse residual | 0.005591 |
| Chunk validation | 0.013261 |
| IK | 0.142528 |
| Execute overhead | 0.028087 |
| Settle/physics | 2.896470 |
| Decision-to-observation total | 136.311728 |
| Episode total | 136.653008 |

The episode was provider/model-bound (97.2% of wall time); provider timing
includes relay and network latency. Relay usage metadata totals 93,761 input,
571 output and **94,332 total tokens**. Usage is provider-reported; no private
reasoning or raw wire capture is stored.

Final offline regression: **313 tests PASS; S1/S2 48/48; Ruff, format, S5 mypy,
compilation and diff checks PASS**. The quality record and credential-content
audit are stored with this episode. Initial, per-decision, grasp, lifted,
transport and final montages are retained, alongside structured actions,
observations, evaluator metrics and timing. One successful bounded episode
supports this simulation checkpoint; it does not establish a general success
rate or physical reliability.

**S5A PASS; S5B attempt #1 PARTIAL; October 10 invocation BLOCKED; October 11
resumed post-fix replication PASS; S5 PASS / FROZEN.** Robot connections,
robot commands, serial connections and physical-camera connections were zero.
This resumption transmitted ten model requests and made no further live
attempt. No S6/E0 work or physical execution was started, and nothing was pushed.
