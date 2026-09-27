# S4 — Three-camera observation parity

Scope: simulator-side logical RGB observation parity on the frozen S1/S2 base
`88d93cf686ba0c67afe06b163aa7fb48d65f1a07`. The implementation lives on
`research/astra-mujoco-s4-cameras` in the isolated `nova_mujoco_s4` worktree.
There is no S3 dependency, hardware access, or S5/Astra integration.

## Verified physical reference

The current target profile is the Phase 5A mixed-resolution camera contract.
Read-only source inspection checked `camera_inventory.json`, `physical_smoke.json`,
the three saved JPEGs, adapter configuration and observation code, and
`robocurve/camera_live.py`. Exact paths, file hashes, and repository revisions are
in [s4_source_audit.json](../../configs/s4_source_audit.json).

| Canonical name | Existing physical identity | Verified decoded RGB | Evidence/view guidance |
|---|---|---|---|
| `front_rgb` | icSpring hub port 3.1, video-index0 | 640×480, uint8 RGB | External front view of arm, tabletop, blue/black objects and red container region |
| `right_rgb` | icSpring hub port 3.2, video-index0 | 640×480, uint8 RGB | External oblique view; arm toward the left, object and red-container regions visible |
| `wrist_rgb` | SN0002 1080P USB Camera, video-index0 | 1280×720, uint8 RGB | Close tabletop/manipulation view; exact hand-eye transform unmeasured |

Phase 5A configured all three at 30 FPS and recorded roughly 30.1 raw JPEG FPS.
This is historical camera-only evidence, not a new physical acquisition. It does
not establish continuous physical performance or synchronized sensor exposure.

X-Trainer Phase 5D separately recorded all three at 640×480. Its raw labels map
`front → wrist_rgb`, `left → front_rgb`, `right → right_rgb`. Those labels and
the all-640×480 recording profile are not the canonical Phase 5A model-facing
contract. S4 keeps exactly `front_rgb`, `right_rgb`, `wrist_rgb`, in that order.
There is no `top_rgb`, legacy `table` slot, depth, or physical camera backend.

## Configuration and calibration classification

[s4_cameras.json](../../configs/s4_cameras.json) contains all camera transforms,
resolutions, vertical FOVs, roles and parameter-by-parameter provenance. Every
camera field is classified as `VERIFIED_FROM_EXISTING_EVIDENCE` or
`SIMULATION_ENGINEERING_PARAMETER`. The renderer parameters are engineering
values. Intrinsics, distortion, camera-to-base and wrist hand-eye transforms,
and sensor exposure timing remain unknown.

| Sim camera | Parent | Position in parent (m) | Vertical FOV | Resolution |
|---|---|---|---:|---|
| `front_rgb` | world | `[-1.00, -0.15, 0.40]` | 55° | 640×480 |
| `right_rgb` | world | `[-0.32, 0.95, 0.38]` | 55° | 640×480 |
| `wrist_rgb` | Link6 | `[0.00, 0.045, 0.035]` | 65° | 1280×720 |

Both external cameras are fixed, low, downward-oblique engineering placements.
Their orientations are stored as explicit unit wxyz quaternions; the front is
aimed toward world `[-0.12,-0.10,0.24]`, the right toward `[-0.16,-0.08,0.22]`.
The selected world coordinates are not recovered physical extrinsics.

The wrist quaternion is `[0,1,0,0]` (180° about local X). MuJoCo camera axes are
X right, Y up, optical direction **−Z**, so the wrist camera looks along Link6
**+Z**. Its translation and rotation are explicit **engineering assumptions**.
It attaches to Link6, not to a claimed calibrated gripper/hand-eye frame. The
physical camera's logical near-manipulation role is supported; rigid mounting
accuracy is not established by the saved photographs.

World and base are distinct: the validated model retains its world-to-base
translation of `[0,0,0.03] m`. Camera poses and fixture positions are world-frame
metres; `eef_state` remains base-relative. Rig clipping is 0.01–10 m, converted
to MuJoCo's extent-scaled near/far settings. MSAA=4, max render geoms=10000.
No image resizing or USB latency simulation is applied.

## Architecture and usage

`NovaMujocoCameraRig` composes cameras into an in-memory render copy of the
simulator's standalone MJCF. It never writes or alters `nova2.xml`, joint axes,
link transforms, masses, collision geometry, or the source simulator state.
The existing simulator and Cartesian bridge are unchanged.

```python
# Set MUJOCO_GL=egl before Python imports MuJoCo.
from nova_cameras import NovaMujocoCameraRig
from nova_simulator import NovaMujocoSimulator

with NovaMujocoSimulator() as sim:
    sim.reset(q_start_rad)
    with NovaMujocoCameraRig(sim, controlled_frame="link6") as rig:
        observation = rig.observe(instruction="Scripted observation")
        front = observation.images["front_rgb"]  # (480, 640, 3), uint8 RGB
        wrist = rig.render("wrist_rgb")          # independent CameraFrame
        debug_rgb = rig.render_debug()           # optional axes/frustum overview
```

The library creates no default fixture objects. The evidence CLI opts into
`fixtures/s4/visual_fixtures.xml`: a static tabletop, blue box, black box and
red target region with collisions disabled. These are camera test fixtures;
the red region is not a grasp/container simulation. S3 owns production scene
objects and physics after merging. S4's renderer copies the complete generalized
state, including extra free joints and mocap state, so later source scenes can
add objects without changing camera code. An added moving-object regression
tests this boundary.

The compositor currently requires a standalone MJCF; XML includes fail explicitly.
Use a resolved/exported standalone production scene on a later merge. Source
model geometry and materials are assumed static for the rig lifetime; rebuild
the rig after model-level changes. Source qpos/qvel/mocap state can change
between observations. Use a rig on one thread and do not mutate the source
while a snapshot is being copied. The copied state then stays frozen throughout
all three sequential render calls.

## Observation and freshness mapping

`SimObservation` mirrors the public Inspect Robots field names without importing
the framework: `images`, `state`, `instruction`, `image_times`, `state_time`,
`extra`. In future S5, a thin wrapper can construct:

```python
# Documentation only: S4 never imports or calls Inspect Robots or its agent.
Observation(images=sim_obs.images, state=sim_obs.state,
            instruction=sim_obs.instruction, image_times=sim_obs.image_times,
            state_time=sim_obs.state_time, extra=sim_obs.extra)
```

`state["eef_state"]` is seven values: absolute base x/y/z metres, reset-relative
yaw/pitch/roll radians, inactive gripper zero. Orientation uses the inverse of
`Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ R_reference`, with explicit rejection near
the relative Euler singularity. It is computed from the same copied state as
the images, not filled with synthetic zeros after arbitrary joint movement.
`joint_pos` is the six-joint radians vector. The controlled frame is explicit.

Each successful `observe()` gives all three images the same simulation state
time, state hash, and increasing `frame_generation`. A new generation is also
allocated for every independent render, even if the state is static. Resetting
the simulator may rewind simulation time; generation does not rewind within
that rig. Creating a new rig starts a new stream. The S5 wrapper must track rig
lifecycle rather than comparing generation values across unrelated rigs.

Rendering does not step the simulator or sleep. `image_times` and `state_time`
are **simulation seconds**, not host monotonic acquisition times. Per-camera
metadata under `extra["nova_mujoco"]["camera"]["frames"]` includes simulation
time, generation, state hash, world pose, and host monotonic render start/end.
Host render timing is diagnostic only. A failed required camera raises and no
partial observation is returned. A batch generation is committed only after
all requested views succeed. Images and state arrays are owned, read-only copies.

Do not feed simulation timestamps to the real adapter's host-time age guards.
There is no simulated exposure, transport buffering, USB delay or camera noise.
Same-state coherence is stronger than physical exposure synchronization and
must not be used as evidence about the latter.

## Real/sim contract table

| Field | Real Phase 5A | Sim S4 | Status |
|---|---|---|---|
| Logical names | front_rgb, right_rgb, wrist_rgb | Same exact names, stable ordering | MATCH |
| Resolution | 640×480, 640×480, 1280×720 | Same native render/output dimensions | MATCH |
| RGB dtype/layout | Decoded HWC uint8 RGB | HWC uint8 RGB, top-left origin | MATCH |
| Fixed/wrist attachment | External front/right roles; wrist/near manipulation; exact mounts not calibrated | Fixed world front/right; rigid Link6 wrist | APPROXIMATE |
| FOV / intrinsics | Unmeasured | Pinhole vertical 55°/55°/65°, square pixels, centered principal point | UNKNOWN physical / APPROXIMATE sim |
| Extrinsic / hand-eye | Unmeasured | Explicit engineering transforms | APPROXIMATE |
| Timestamp semantics | Host complete-JPEG receive time; not exposure | Frozen simulation-state time; separate host render times | APPROXIMATE logical mapping; clock domains differ |
| Generation/freshness | Reader generation+sequence and post-barrier host receive checks | Monotonic rendered-batch generation, same state snapshot; no stale cache | MATCH logical new-frame property; APPROXIMATE timing |
| Distortion | Unmeasured | Ideal undistorted pinhole | UNKNOWN physical / DEFERRED modeling |
| Exposure timing | Unsynchronized/unknown sensor exposure; host timing only | No physical exposure; coherent frozen state | DEFERRED physical matching |

## Validation and reproduction

Use the existing environment; no package download, hardware environment, or
DISPLAY is needed:

```bash
cd /home/dsa/project/dobot/.worktrees/nova_mujoco_s4
PY=/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python
MUJOCO_GL=egl "$PY" -m pytest -q
MUJOCO_GL=egl "$PY" run_s4_camera_validation.py \
  --repetitions 30 --debug
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" -m compileall -q -x 'DOBOT_6Axis_ROS2_V4/.*' .
"$PY" compare_pose.py real_nova2_samples.json --orientation rpy
git diff --check
```

Tests set EGL before MuJoCo is imported unless another backend is explicitly
selected. No GUI fallback is used. Headless context creation failures are
reported; tests do not silently skip rendering. EGL is verified on this host;
OSMesa is selectable but not claimed tested here. The rendering code uses
public GLContext/MjrContext APIs and explicitly frees both, including on errors.

The runner refuses existing evidence directories. It saves `camera_config.json`,
`camera_roles.json`, `validation.json`, three native RGB PNGs, a montage, two
movement montages, and optional `camera_debug.png`. The overview shows arm,
fixtures, camera frustums (yellow), and axes (RGB=X/Y/Z; optical direction −Z).
No debug overlays enter the model-facing images. JSON records model/source
hashes, MuJoCo/backend/OpenGL versions, dimensions, camera poses, initial and
changed states, static/reset pixel identity, ground-truth projected fixture
points, RGB patch checks, and timings. Timings are host measurements, never
used to advance simulation time. Tests additionally verify independent views,
observation failure, configuration errors, and lifecycle cleanup.

The offline audit hook rejects Python socket creation and physical camera/serial
device opens; it is extra instrumentation, not an OS sandbox. Graphics driver
access for offscreen rendering is allowed. No V4L2, camera reader, serial,
robot network, model-provider, or Inspect Robots runtime dependency is present.

## Remaining gaps and stop boundary

This is logical observation parity, not pixel-perfect rendering or a calibrated
digital twin. Room background, photometry, exposure, noise, distortion, exact
view angles, gripper visuals, contacts, and physical timing are absent or
approximate. Geometry/FK retain their separate S1/S2 evidence. Image identity
is tested on the recorded GL stack; cross-driver bit identity is not promised.

S4 prepares the input boundary for S5. It does not answer State Gap, Effect Gap,
Candidate Interface, constraint, or sim-to-real behavioral questions. No Astra,
OpenAI, agent, physical-camera, robot or serial request is made. Stop after the
S4 checkpoint; S3 merging and S5 require a later task.

## Checkpoint — September 27, 2026

**S4 PASS for logical three-camera observation parity.** All ten exit gates
are met: exact roles; fixed external cameras; rigid wrist attachment; HWC RGB
uint8; no extra canonical camera; coherent image/state snapshot; parameter
provenance; preserved S1/S2 tests; no hardware access; saved rendered evidence.

The evidence run is
[`20260927T063657Z`](../../evidence/sim_s4/20260927T063657Z/summary.md),
generated from clean committed revision
`093e79c555fa22d9b23b997e1d0a6d5687bdae08`. The final checkpoint commit adds
evidence and documentation only; the source revision and hashes remain explicit.

- [Validation JSON](../../evidence/sim_s4/20260927T063657Z/validation.json)
- [Camera configuration](../../evidence/sim_s4/20260927T063657Z/camera_config.json)
- [Role mapping](../../evidence/sim_s4/20260927T063657Z/camera_roles.json)
- [Three-view montage](../../evidence/sim_s4/20260927T063657Z/montage.png)
- [Frustum and axes overview](../../evidence/sim_s4/20260927T063657Z/camera_debug.png)
- [Quality gates and frozen-file comparison](../../evidence/sim_s4/20260927T063657Z/quality_gates.json)

The renderer was MuJoCo 2.3.7, EGL, Mesa 25.2.8 llvmpipe (LLVM 20.1.2),
OpenGL 4.5. This is software rendering on the current host. Thirty warmed
repetitions per measurement, including state copy and image readback, gave:

| Mode | Frames or observations per host second | Mean milliseconds per state |
|---|---:|---:|
| front only | 54.75 FPS | 18.26 |
| right only | 53.28 FPS | 18.77 |
| wrist only | 70.71 FPS | 14.14 |
| all three | 19.27 coherent observations/s | 51.90 |

These are scene-specific approximate throughput figures. The wrist sees less
arm geometry, so its larger resolution need not make it the slowest view.
At about 52 ms per image triplet, 100 observation-only iterations take roughly
5.2 seconds after startup on this scene. Future S5 must measure actual inference
latency before claiming a model-bound or render-bound closed loop.

All three static and reset renders repeated exactly on this GL stack. New rig
construction also passed image identity tests. In two changed joint states,
front/right position and rotation changes were exactly zero; the wrist moved
29.57 mm and 22.98 mm respectively. Images changed for all three roles. All nine
fixture projection/color checks passed at each of the three tested states.
Full per-frame state hashes, poses, generations and timings are in the JSON.

Final gates: **102 tests passed in 9.25 s** (68 unchanged baseline tests plus
34 S4 tests); Ruff check and format passed (27 Python files); compilation and
diff check passed. The frozen MJCF, simulator, Cartesian bridge, FK code, raw
real samples, and every baseline test are byte-identical to the frozen base.
The real-sample FK rerun remains position RMS/max **0.010882/0.012292 mm** and
orientation RMS/max **0.000472/0.000538 degrees**.

The evidence run recorded zero blocked I/O attempts and:

```text
robot_connections = 0
robot_commands = 0
serial_connections = 0
physical_camera_connections = 0
openai_requests = 0
```

The test suite additionally makes deliberately prohibited calls to prove the
audit hook raises before socket/device creation. Those negative tests do not
open physical endpoints. No S3 worktree or other repository was edited, and no
push occurred. Stop here; S5 was not started.
