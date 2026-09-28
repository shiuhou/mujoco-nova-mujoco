# S5 — official Astra agent to unified MuJoCo

S5 validates the model-facing closed loop in simulation.
It does NOT authorize autonomous physical Nova execution.

Current live status: **S5A NOT YET RUN; S5B BLOCKED pending S5A PASS**.
The offline implementation is committed at
`2689e4a772d6eb8ec48bc5df0aba220944f916d0` (OFFLINE_S5_HEAD). The operator
has supplied a runtime credential source and confirmed `gpt-6-astra` through
the Responses relay at `https://sub.scitrace.cc/`. No live request preceded
the offline rollback freeze.
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
as `OPENAI_API_KEY` without displaying or persisting it, and pass the confirmed
model and endpoint explicitly. The credential source is not committed here.

```bash
# Only after both operator-supplied environment values and current gates exist:
MUJOCO_GL=egl .venv-s5/bin/python run_s5a.py --mode live \
  --model gpt-6-astra --base-url https://sub.scitrace.cc/ \
  --quality-gates /tmp/nova-s5-quality-UNIQUE/quality_gates.json \
  --max-llm-calls 2 --max-episode-steps 32 --max-decisions 2 \
  --max-wall-time 90 --output evidence/sim_s5a/UNIQUE-live
```

An explicit `--model` overrides `INSPECT_ROBOTS_MODEL`. Missing credentials,
missing model or stale quality gates return exit code 2 before provider/network
construction. The live factory accepts no injected fake/replay transport.
Offline tests inject mock transports independently and never access OpenAI.

## Results and remaining boundary

Current evidence paths and final quality counts are recorded in the completion
record below. The live exit gate remains unfulfilled even when all offline
checks pass. S5B must wait for a live S5A PASS and then use the exact benchmark
instruction, autonomous official-agent decisions, and S3 geometric/contact
evaluation. It must not feed the scripted S3 trajectory or hidden coordinates
to the model. No S5B evidence or success claim is produced here.

Future real execution remains separately gated by `pre_check`, approver,
one-use authority, bounded MovL, settle and fresh physical observation, with
the physical E-stop as the primary emergency stop. S5 implements none of that
physical execution and removes none of those requirements.

Existing limitations remain: primitive assets, unmeasured TCP, engineering
camera extrinsics/FOV, simplified gripper/contact dynamics and prescribed arm
references. Live API availability, exact Astra identifier, latency, visual
competence and autonomous pick-and-place success are unvalidated.

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
