# S3×S4 production manipulation and observations

**Integration: PASS. S5 readiness: READY for subsequent offline policy-integration
work. S5 has not started.** No provider, physical robot, serial device or physical
camera was accessed. **THIS IS NOT A CALIBRATED DYNAMICS DIGITAL TWIN.**

## Inputs, branch order and ownership

| Input | Frozen branch | Commit |
|---|---|---|
| S1/S2 | `research/astra-mujoco-embodiment` | `88d93cf686ba0c67afe06b163aa7fb48d65f1a07` |
| S3 | `research/astra-mujoco-s3-manipulation` | `dea5c927e52162d178a3e97d97d6541771dbcab0` |
| S4 | `research/astra-mujoco-s4-cameras` | `269f34681a7fee0da3297df3f6df6f551ccd351e` |

The isolated integration worktree is
`/home/dsa/project/dobot/.worktrees/nova_mujoco_integration` on
`integration/astra-mujoco-s3-s4`. It started at the exact S1/S2 base, advanced
to frozen S3, merged frozen S4, then added integration fixes/tests. No merge
conflicts required discarding either phase. The reviewed, validated result is
merged locally into `research/astra-mujoco-embodiment` in
`/home/dsa/project/dobot/nova_mujoco`. No push; S3 and S4 branches/worktrees remain
at their frozen commits. Root pins and other repositories are untouched.

S3 owns production scene specs, objects, gripper, contact physics, action
execution and task success. Its runtime/model/builder/script and all S3 tests
are unchanged. S4 owns `NovaMujocoCameraRig`, cameras, RGB/state bundles, state
hash, frame generations and render timing. Integration changes only the camera
composition/state seam, adds an explicit production camera profile, a validation
runner and dedicated tests; all prior phase tests remain unchanged.

## Production scene remains S3

| Object | Dimensions | Behavior |
|---|---|---|
| `blue_block` | 60 × 60 × 60 mm | dynamic manipulation target |
| `black_block` | 50 × 50 × 50 mm | independent dynamic distractor |
| `red_box` | 150 × 150 × 80 mm | static open-top container, five collision pieces |

Walls are 5 mm, explicitly a **SIMULATION ENGINEERING PARAMETER**. The interior
is 140 × 140 × 75 mm. All three assets remain `temporary_primitive`. There is
no hidden lid, object weld or episode object teleport. Contact forces produce
grasp, lift, transport and release. All original masses, friction, solver
settings, reset poses, joint references and geometric placement checks are
preserved. See [S3](s3_manipulation.md) for exact poses and parameters.

The gripper retains **0 closed / 1 open**, 95 mm maximum requested gap,
flange +Z 197 mm nominal grasp center and **measured_tcp=false**. Measured
aperture during a closed-command grasp is approximately 59.6 mm because the
object prevents full closure. Normalized measured aperture is therefore about
0.6275 rather than the command 0. Tiny numerical endpoint excursions remain
visible; observations do not silently clamp away contact compliance.

`fixtures/s4/visual_fixtures.xml` is never passed to the production rig. Tests
verify no `s4_*` geoms, identical production geom counts/IDs, exact dimensions,
five container pieces, and identical compiled joint/body/geom/actuator/equality
arrays between physics and camera models.

## Integration fixes and reusable API

1. The camera compositor expands local MJCF includes **in memory**, preserving
   child order and source asset paths. Paths are relative to the main MJCF,
   matching the installed MuJoCo behavior. Duplicate/cyclic/missing-file includes
   fail explicitly. No flattening output overwrites S3's modular files.
2. The camera rig inherits `sim.config.default_frame` when no frame is explicitly
   supplied: S1/S2 stays `link6`; S3 uses `nominal_grasp_center`.
3. The seventh `eef_state` element reads the optional source `get_gripper_state()`
   API's measured normalization. S1/S2 without that API stays inactive zero.
   Camera code does not import S3 or redefine the gripper's task semantics.
4. A frame with wrong shape/dtype/layout or no array fails before generation
   commit. A middle-camera exception produces no partial observation or new
   generation; retry renders a complete coherent bundle.
5. `diagnostic_geom_ids()` uses native MuJoCo segmentation for visibility checks,
   outside model-facing RGB observations. It restores render flags, does not
   advance observation generation and cannot alter physics state.

```python
# Set MUJOCO_GL=egl before importing MuJoCo.
from manipulation_simulator import ManipulationSimulator, ManipulationChunkExecutor
from nova_cameras import NovaMujocoCameraRig
from run_s3_s4_integration import PRODUCTION_CAMERA_CONFIG

with ManipulationSimulator() as sim:
    sim.reset()
    executor = ManipulationChunkExecutor(sim)
    try:
        with NovaMujocoCameraRig(sim, config_path=PRODUCTION_CAMERA_CONFIG) as rig:
            observation = rig.observe()
            p = sim.get_tool_pose().position_m
            result = executor.execute_chunk([[*p, 0, 0, 0, 1]])
            fresh_observation = rig.observe()
    finally:
        executor.close()
```

Use one owner/thread; source model geometry/configuration is static for a rig's
lifetime. Rebuild the rig after changing model assets or camera configuration.
The generalized state, including both free objects and fingers, is copied into
the independent render data. `mj_forward` runs only on that copy. Rendering
never calls source `step()`, `reset()`, or a physical device backend.

## Camera contract and production profile

Canonical image names and order are exactly:

| Name | Native HWC shape | Type/order | Attachment |
|---|---|---|---|
| `front_rgb` | (480, 640, 3) | uint8 RGB, top-left origin | fixed world |
| `right_rgb` | (480, 640, 3) | uint8 RGB, top-left origin | fixed world |
| `wrist_rgb` | (720, 1280, 3) | uint8 RGB, top-left origin | rigid Link6 |

Arrays are contiguous/read-only, with no policy-image resizing or depth channel.
Montages alone downscale images for human review; full native PNGs are also
saved. `eef_state` is seven finite values: base-frame x/y/z metres,
reset-relative yaw/pitch/roll radians, measured normalized aperture. It targets
the same nominal grasp-center frame as S3 action execution. Nonzero orientation
actions remain outside the frozen S2 subset; this integration does not add them.

`configs/s4_cameras.json` remains the exact frozen standalone profile. The
production runner explicitly uses `configs/s3_s4_cameras.json`. Front/right
poses, FOV, resolutions, clipping and rendering settings are unchanged. The
original wrist placement was partly occluded by S3's simplified palm, so the
production profile changes only its **engineering** Link6 translation and
rotation: position `[0, .100, .100]` m, optical direction aimed at nominal
`[0,0,.197]` m. The exact unit quaternion and parameter-level provenance are
in that JSON. This is an explicitly uncalibrated simulation mount, not a measured
hand-eye transform. The wrist still moves rigidly with Link6; no world-following
or look-at tracking is used during an episode.

Both fixed cameras see all production objects throughout the tested sequence.
The near wrist view sees the blue block throughout grasp/lift/transport/release;
it need not include the remote black distractor or red box at every pose.
Native segmentation associates visible pixels with the real S3 geom IDs and
then checks their RGB colors, avoiding false evidence from table shadows or
non-contact camera fixtures. Occlusion is measured rather than inferred solely
from projected centers.

## Freshness, coherence and failure handling

Each `observe()` snapshots once, then renders all three images sequentially
from that copied state. Their `state_hash`, simulation time and frame generation
match each other and the corresponding state bundle. Host monotonic render
start/finish/elapsed values are separate metadata. These are simulated snapshot
times, not physical exposure synchronization or sensor timestamps.

Generations increase only after a complete successful render batch. They advance
even when the source is unchanged, while hashes/pixels then remain identical.
Each manipulation phase in this checkpoint has a newer simulation time, changed
state hash, new generation and changed images. After an injected `right_rgb`
failure, the generation stays at 8; successful retry produces generation 9.
Previously returned images remain unchanged. Missing/malformed frames are also
rejected in parameterized tests.

The integration runner reports PARTIAL and an observation-failure reason if
rendering fails; it cannot publish a successful manipulation/observation proof
using stale images. Output directories are never overwritten. The deliberate
failure probe is labeled as injected and isolated from the successful phase
observations.

## Proven sequence and results

`run_s3_s4_integration.py` calls the **unchanged** `scripted_episode()` with a
checkpoint callback. Observations are captured after reset, approach, grasp,
lift, transport, lowering, release and final retreat. The required path is:

```text
reset production scene -> native RGB + eef_state
-> contact grasp -> fresh observation
-> lift -> fresh observation
-> transport -> fresh observation
-> release into red box -> fresh observation
-> retreat/settle -> fresh observation -> inside_target=true
```

Every phase passes native camera/state contract, source-state preservation,
fixed-camera invariance, rigid wrist, freshness and production visibility checks.
Before and after each observation/diagnostic, MuJoCo `mjSTATE_INTEGRATION`
(including warm starts/applied forces), qacc and task metrics are checked.
An independent complete episode without camera rendering is compared with the
observed episode: **the full physics result/waypoints/trajectory are exactly
equal on this runtime**.

Nominal outcome: grasp/lift/transport/release/settled placement PASS,
`inside_target=true`, peak lift 148.623 mm, maximum slip 4.842 mm, distractor
maximum displacement 0.0170 mm (vertical settling only), final aperture 95 mm.
The episode takes 25.882 simulated seconds and retains all S3 contact logic.

## Tests and evidence

The merged pre-fix baseline passed all 153 prior tests. The integration adds
23 tests: **176 total = 68 S1/S2/prior + 51 S3 + 34 S4 + 23 integration**.
Ruff lint/format, Python compilation, diff checks, the 48/48 S1/S2 micro-move
benchmark and recorded-sample FK comparison also pass. No prior test is weakened
or rewritten. The original phase branches and evidence remain frozen.

Final evidence:
[`evidence/sim_s3_s4/20260927T162753Z/`](../../evidence/sim_s3_s4/20260927T162753Z/).

- `validation.json`: phase gates, unchanged physics, failure probe and zero I/O.
- `integration_result.json`: complete S3 episode, final metrics/object states.
- `observation_metadata.json`: eight phase observations, hashes/generations,
  eef/gripper/object states, geometry visibility, transform errors and timings.
- `environment.json`: exact source hashes/revision, MuJoCo/Python/NumPy/platform/
  OpenGL and scene/camera configs; frozen input commits are in the integration result.
- `quality_gates.json`: actual test/lint/compile/benchmark/FK results, source
  invariants and unchanged frozen-branch checks.
- `initial_montage.png`, `grasp_montage.png`, `lifted_montage.png`,
  `transport_montage.png`, `release_montage.png`, `final_montage.png`, plus
  native-resolution three-camera PNGs at every phase.

Run from either the integration worktree or unified repository:

```bash
MUJOCO_GL=egl /home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python \
  run_s3_s4_integration.py --output evidence/sim_s3_s4/<new-unique-name>
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m pytest -q
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m ruff check --exclude DOBOT_6Axis_ROS2_V4 .
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m ruff format --check --exclude DOBOT_6Axis_ROS2_V4 .
```

The tests select EGL before importing MuJoCo. The audit guard blocks Python
sockets and physical camera/serial nodes; GL rendering uses only the graphics
stack. Guard tests deliberately verify blocked attempts, while the integration
runner itself records zero attempted network/device access. The guard is a
Python audit backstop, not an OS security sandbox.

## Limitations and S5 boundary

- Primitive assets, simplified fingers, nominal unmeasured TCP and engineering
  dynamics/contact/arm-reference parameters remain exactly S3's limitations.
- Camera poses/FOVs are uncalibrated; no physical hand-eye, distortion, lighting,
  exposure, noise, latency, room appearance or sim-to-real fidelity claim.
- The wrist frame is rigid and useful for this scripted task, not a measured
  physical camera mount. Natural occlusion remains; not every object is visible
  in every view.
- Rendering is serial and requires EGL/OSMesa; static image equality and exact
  physics comparison are scoped to the recorded stack, not all GPU/platforms.
- General arbitrary orientation actions and learned-policy task success are not
  validated. No provider/agent dependency, API call or S5 loop was added.

**S5 READY** means the offline production scene, action executor, contact task
metrics and coherent fresh observation interface are integrated and tested as a
starting point for later S5 work. It does not mean S5 is implemented, or that
any live hardware, policy/provider integration or real-world safety is ready.

```text
robot_connections = 0
robot_commands = 0
serial_connections = 0
physical_camera_connections = 0
openai_requests = 0
```
