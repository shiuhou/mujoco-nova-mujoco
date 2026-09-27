# S3 manipulation: benchmark_v0

**Status: PASS for this offline manipulation-interface checkpoint.**

**THIS IS NOT A CALIBRATED DYNAMICS DIGITAL TWIN.**
This S3 checkpoint validates manipulation-interface behavior and a reproducible
MuJoCo benchmark environment. No result is physical robot safety evidence.

## Scope and frozen base

- Repository: `/home/dsa/project/dobot/nova_mujoco`.
- Isolated worktree: `/home/dsa/project/dobot/.worktrees/nova_mujoco_s3`.
- Branch: `research/astra-mujoco-s3-manipulation`.
- Exact base: `88d93cf686ba0c67afe06b163aa7fb48d65f1a07`.
- Original `nova2.xml`, `nova_simulator.py`, `cartesian_bridge.py`, S1/S2 tests,
  FK sources, calibration gate, and prior evidence remain unchanged.
- No S4/S5 implementation, other repository changes, push, or automatic merge.

## Architecture and reusable boundary

`benchmark_scene.py` owns immutable `BenchmarkObjectSpec`, `GripperSpec`,
`PhysicsSpec`, `ScriptSpec` and `BenchmarkSceneSpec` values. The checked-in
`scenes/benchmark_v0/scene.xml` composes `arm.xml`, `gripper.xml` and `objects.xml`.
`python benchmark_scene.py` regenerates them deterministically from the specs and
frozen `nova2.xml`. The arm include preserves all six axes, link transforms,
joint limits, visual meshes, collision proxy shapes, Link6 and nominal tool
sites. Its only arm additions are prescribed-motion armature and the gripper
include at Link6. Regeneration/FK equivalence are automated tests.

`ManipulationSimulator` extends the frozen S1 simulator in explicit
`kinematic_arm_with_dynamic_objects` mode. It provides:

```python
from manipulation_simulator import ManipulationSimulator, ManipulationChunkExecutor

with ManipulationSimulator() as sim:  # Loads the production scene.
    sim.reset()
    blue_pose = sim.get_object_pose("blue_block")  # world metres, rotation matrix
    blue_state = sim.get_object_state("blue_block")  # also velocities/quaternion
    sim.command_gripper(1.0)
    sim.hold(0.2)  # Fixed simulation ticks; no host waiting.
    gripper = sim.get_gripper_state()
    metrics = sim.compute_manipulation_metrics()
    executor = ManipulationChunkExecutor(sim)
    p = sim.get_tool_pose().position_m  # nominal grasp center, base metres
    result = executor.execute_chunk([[*p, 0, 0, 0, 1]])
    executor.close()
```

Create the executor after reset, or use its inherited `reset_reference()` after
a subsequent reset. `get_robot_state()` retains the S1 pose/joint state seam;
gripper and semantic objects are separately queryable. The scene has no required
camera dependency. Renderer code is confined to the benchmark utility.

`ManipulationChunkExecutor` reuses the existing IK solver and S2 result records.
It accepts absolute base-frame metres and the same seven-value action order
`[x,y,z,yaw,pitch,roll,gripper]`. As in S2, only zero orientation deltas are
supported, preserving reset orientation; nonzero deltas are rejected. Gripper
is active only in the S3 executor. Waypoint displacement is limited to 0.020 m;
the script subdivides moves into at most 0.010 m requests. A failed waypoint
stops the remainder. Validation and IK precede gripper/arm actuation.

IK runs on a separate frozen-model data instance. Trial joint poses cannot
change live free-object coordinates, contact warm starts, or episode time.

## Temporary primitive objects and exact initial poses

These are **temporary_primitive** assets, not CAD-derived geometry. Object
dimensions below are the operator's source of truth. Body origins are the
centers of the outer dimensions, and all initial quaternions are `[1,0,0,0]`
(wxyz). Positions are **world metres**; the frozen base origin is at world
`[0,0,0.030]`, so subtract 0.030 from world z for base-relative actions.

| Semantic body | Role | Outer dimensions (m) | Initial world position (m) | Dynamic |
|---|---|---|---|---|
| `blue_block` | blue manipulation target | 0.060 × 0.060 × 0.060 | (-0.300, -0.150, 0.060) | yes |
| `black_block` | black distractor | 0.050 × 0.050 × 0.050 | (-0.480, -0.280, 0.055) | yes |
| `red_box` | red placement target | 0.150 × 0.150 × 0.080 | (-0.300, 0.080, 0.070) | no |

The support's top is world z=0.030 m. Its half-size is (0.65, 0.50, 0.02) m and
center is (-0.20, 0, 0.01) m. Support size/placement and all scene placements are
SIMULATION ENGINEERING PARAMETERS. The two dynamic blocks are separate
free-joint bodies; no object equality/weld is present.

The box has exactly five geoms: bottom, left, right, front and rear. Its 0.005 m
wall thickness is a configurable **SIMULATION ENGINEERING PARAMETER**. The
derived clear interior is 0.140 × 0.140 × 0.075 m, with local bounds
`[-.070,-.070,-.035]` to `[.070,.070,.040]`. In world z, the inner floor is
0.035 and rim is 0.110. The opening accommodates both the 60 mm block and the
opened gripper pads; no roof/invisible collision spans it. The builder and
placement metric both use `interior_bounds_local_m`, not duplicated dimensions.

### Reachability and initial validity

The initial joint vector is:

```text
[0.1057414219467702, -0.07307772766776358, 1.949250880153336,
 -0.30537314570463386, -1.5707963268520606, 1.6765303917661518] rad
```

It was found using the existing validated IK, seeded from recorded `pose_01`
`[-0.8297,-36.7051,123.6153,1.4609,-88.6983,-89.2010]` degrees, targeting the
nominal grasp center at base `[-.30,-.15,.15]` and a downward flange +Z axis
(`R=diag(1,-1,-1)`). This is a simulator arrangement, not a proposed physical
robot configuration. Every approach, lift, transport and lower waypoint is
verified through that same bounded IK solver. Tests check nonoverlap, separate
distractor, no initial arm contacts, and all successful waypoint residuals.

## Nominal gripper geometry and semantics

- Flange/Link6 -> simplified gripper mount: identity translation and rotation,
  an explicitly constructed simulation frame, not a recovered USD mount.
- Forward axis: flange +Z. `grasp_center` and the unchanged
  `nominal_grasp_center` site are flange `[0,0,0.197]` m.
- **measured_tcp = false**. +197 mm is nominal operator-provided geometry,
  not a precision calibrated TCP.
- Finger joint frames: centers at `(±.006,0,.197)` in the flange frame, with
  outward prismatic axes ±X and ranges `[0,.0475]` m. Each finger pad is
  12 × 40 × 50 mm (engineering approximation). The named `left_fingertip` and
  `right_fingertip` sites mark their inner contact-face centers, symmetrically
  displaced ±half the aperture from `grasp_center`.
- A joint equality couples the two finger displacements. It is not attached
  to an object. Compliance permits small temporary asymmetry under contact.
- **0.0 = closed (0 mm requested gap); 1.0 = open (95 mm requested gap)**.
  Intermediate values map continuously. NaN, infinity, nonscalar and values
  outside [0,1] are rejected; there is no policy-output clamping.
- The actuator setpoint aperture slews at 0.060 m/s. Commands, bounded
  setpoint, and measured gap are distinct. Dynamic fingers can be blocked
  by objects: a closed command during grasp measures about 59.6 mm.
- Measured gap/normalization reports the simulated coordinates without hiding
  solver compliance. Tiny numerical excursions around travel endpoints are
  possible; the command endpoints remain exactly 0 and 0.095 m.

The existing workspace/history contains official extracted finger candidates
under `assets/xtrainer_gripper`, originating at pinned X-Trainer dataset
revision `02f008252001f654e28dee40d615c713ff79935a`; extraction is in the baseline
history. Their local meshes are available, but the USD assembly transforms do
not identify the Nova flange. S3 deliberately uses the authorized simplified
geometry and leaves the physical calibration gate unchanged. No assembly
transform or new CAD provenance is invented.

## Physics and exact stepping

All entries in this section are **SIMULATION ENGINEERING PARAMETERS**, not
measured physical values. The complete configuration is recorded in
`environment.json`; the specifications centralize tunable object/physics values.

| Parameter | Value |
|---|---|
| Simulation timestep | 0.002 s |
| Integrator / cone / solver iterations | implicit / elliptic / 100 |
| Gravity | (0,0,-9.81) m/s² |
| Contact solref / solimp | (.008,1) / (.95,.99,.001) |
| Block masses | blue .10 kg; black .08 kg |
| Static box mass metadata | .30 kg total; static body has no dynamics |
| Object/support friction | sliding .9, torsional .005, rolling .0001 |
| Finger friction | sliding 1.0, torsional .005, rolling .0001 |
| Finger mass / damping / position gain | .08 kg each / 8 N·s/m / 2500 N/m |
| Finger actuator force limit | ±8 N each |
| Finger equality solref | (.004,1) |
| Prescribed arm joint armature | 1,000,000 kg·m² each, numerical device only |
| Arm reference maximum joint speed | .30 rad/s |
| Script Cartesian duration bound | .040 m/s between requested targets |

The simplified stem mass is .20 kg, palm mass .15 kg. Their dimensions and all
visual/collision additions are centralized in the scene builder. Free-block
inertia is generated from the explicit primitive mass and dimensions. No extra
free-body damping or calibrated density/motor-current model is introduced.

Each tick evaluates the S1 bounded linear joint reference, supplies its arm
velocity `(q_next-q_current)/dt`, slews the gripper setpoint, and calls
`mj_step`. MuJoCo integrates free objects and actuated fingers using contact
forces, gravity and inertia. Afterwards **only the six arm joints** are projected
back to the exact reference and zero endpoint arm velocity; `mj_forward`
refreshes geometry/diagnostics. Integer tick × dt is the simulation clock.
Large armature makes arm response to within-tick contact negligible; recorded
maximum projection correction is about 7e-11 rad. This approximation is
explicit, not a newly validated dynamic arm controller.

Object pose writes occur only through MuJoCo's explicit scene reset/keyframe.
There is no object attachment, teleport, mocap object, hidden grasp force,
or diagnostic weld implementation. Contact parameters are deliberately soft;
the nominal run's maximum contact penetration is about 0.23 mm. Frozen arm
collision boxes are coarse and cannot establish physical collision safety.

## Contact evidence and success semantics

Actual MuJoCo contacts above .05 N identify each finger's touching geoms. Blue
must have bilateral finger contacts, lie in the grasp region, and maintain
relative position with speed below .010 m/s for .15 s and slip at most .008 m.
That produces `HOLDING_CANDIDATE` while still supported on the table.
`HOLDING` additionally requires a lift of at least .040 m from its reset pose,
so closing alone and bilateral on-table contact cannot establish a grasp.
States also include `OPEN`, `CLOSING`, `CONTACT`, `OPENING`, `CLOSED_EMPTY`.

`object_grasped` and `grasp_stable` are episode-history flags; `holding_now` is
the current state. `slip_distance` is the greatest candidate-relative slip
observed during the episode. Current and maximum lift are separate metrics.
Frictionless and empty-close tests explicitly disprove command-only success.

Placement transforms all eight blue bounding-box corners into the red body
frame and checks every coordinate against the spec-derived interior, with
.0015 m tolerance. A center-inside block that straddles a wall, penetrates the
bottom or extends above the rim fails. CAD visuals must preserve this semantic
envelope, or explicitly provide an updated conservative collision envelope.

Success also requires blue-to-box-bottom supporting contact, linear speed
≤.010 m/s and angular speed ≤.10 rad/s for .25 s, opening greater than block
width + .005 m, no finger/blue contact, no active hold, no latched robot fault,
prior verified stable lift and slip within bound. Black's **maximum** displacement
during the episode must be ≤.002 m, not merely its final displacement.

Unexpected arm or tool/environment penetration >.0015 m and MuJoCo numerical
warnings latch a fault and stop the chunk until explicit reset. This contact
abort is a simulation guard, not a continuous collision-safety proof. Failures
carry category, stage and reason: `grasp_fail`, `dropped`, `transport_fail`,
`place_miss`, `release_fail`, `collision_abort`, `IK_fail`, or
`safety/validation_abort`. Approach errors retain `stage=approach` with their
IK/collision/validation category.

## Script, evidence and validation

The explicit script resets, opens, approaches the blue block, closes, checks
bilateral contact, lifts, holds, transports over the red box, lowers, opens,
retreats and checks settled placement. It is not Astra or a learned policy.
Targets derive from object specs and script engineering clearances. The nominal
grasp center visits base z=.030 m at grasp, .180 m during lift/transport and
.038 m at release, preserving its reset rotation throughout.

Checkpoint evidence:
[`evidence/sim_s3/20260927T151644Z`](../../evidence/sim_s3/20260927T151644Z/)
(from the repository root; the files listed below are authoritative).

- `validation.json`: two complete fresh episodes, exact equality on this runtime.
- `benchmark_result.json`: stage outcomes, requested/achieved waypoints,
  IK, structured metrics, failures and simulation time.
- `environment.json`: source commit/hashes, MuJoCo/Python/NumPy/platform,
  timestep, full engineering specs, initial poses/joints/aperture, seed=null.
- `final_object_states.json`: all three semantic object states.
- `contact_summary.json`, `contact_timeline.json`: contact forces/pairs/faults.
- `trajectory.jsonl`: sampled object/gripper/arm states through the episode.
- `quality_gates.json`: actual subprocess results and baseline-integrity checks.
- `initial_scene.png`, `grasp.png`, `lifted.png`, `transported.png`,
  `released.png`, `final_scene.png`: headless free-camera inspection images.

Nominal results: grasp, lift, transport, release and inside-target checks PASS;
peak lift ~.148623 m, maximum slip ~.004842 m, final black displacement
~.00001704 m (vertical settling only), final aperture ~.095 m. All original
68 tests and 51 new S3 tests pass (119 total); Ruff lint/format, compilation,
unchanged 48/48 S1/S2 micro-move benchmark and recorded-sample FK comparison
pass. Quality logs/evidence capture the exact values. The S1/S2 rerun is stored in
`evidence/sim_s1_s2/20260927T151013Z-s3-regression/`. The earlier S3
`20260927T151013Z` evidence is retained; only its inspection view was superseded
to make final placement visible. Physics and episode results are identical.

Test coverage includes dimensions/colors/roles, open-box interior, regeneration,
nominal frames/opening/symmetry, continuous/invalid gripper commands, deterministic
reset, bounded IK/arm execution, stop-on-failure, real contacts, empty closing,
frictionless failure, lift/transport/release, rotated corner containment,
unsupported placement rejection, independent distractor, faults and evidence.

The tested environment is Python 3.11.14 / MuJoCo 2.3.7 / Linux. No
cross-platform or cross-version determinism is claimed. Use the existing
environment documented in README, or an equivalent environment with MuJoCo,
NumPy, pytest, Ruff and PyYAML; PNGs additionally use Pillow and EGL.

```bash
python -m pytest -q
python -m ruff check --exclude DOBOT_6Axis_ROS2_V4 .
python -m ruff format --check --exclude DOBOT_6Axis_ROS2_V4 .
python run_s1_s2_benchmark.py --output evidence/sim_s1_s2/<new-unique-name>
MUJOCO_GL=egl python run_s3_benchmark.py --render --output evidence/sim_s3/<new-unique-name>
python run_s3_benchmark.py --viewer
```

Output directories are never overwritten. Renderer failures are recorded and
do not silently masquerade as generated PNGs. The desktop viewer entry point
is provided; an interactive display was unavailable, while headless rendering
was actually exercised. Generic raw-MJCF stepping is not the prescribed-arm
controller: use the runner for the benchmark or its viewer for a held-arm scene.

## Known limitations and S4 expectations

- Primitive blocks/container and simplified fingers, not official CAD fidelity.
- Nominal uncalibrated tool frame; no physical adapter/gripper polarity claim.
- Numerical prescribed arm and soft-contact parameters, not measured dynamics.
- Scripted axis-aligned top-down task with pinned reset orientation; arbitrary
  orientation and other tasks are not validated by this checkpoint.
- Single-platform/version determinism only, no motor dynamics or hand-eye work.
- Offline contact/force/proxy diagnostics are not physical safety evidence.
- The scene can be composed with a future camera rig. S4 may consume
  `sim.model`, `sim.data`, pose/joint state and the gripper query after stepping.
  S3 does not define `front_rgb`, `right_rgb`, `wrist_rgb` poses/intrinsics,
  observation packaging or calibration. Its free inspection camera has no
  model-facing observation contract. No S4 worktree was modified.

## Future STEP/CAD replacement

Desired flow: original `blue_block.step`, `black_block.step`, `red_box.step` →
pinned deterministic tessellation → visual mesh → simplified collision
geometry → the same semantic object specifications. Conversion is not
implemented until source assets exist.

Record actual source revision/license/hash, units and mesh-to-semantic-body
transform in the spec/asset manifest. Replace the builder's visual-geometry
branch with the derived mesh and retain explicit simple collision shapes and
the container's unobstructed interior. Keep semantic names/body frames, task
dimensions/envelopes, roles, object APIs, reset, gripper commands, seven-value
action contract and placement metric semantics. Update asset-specific geometry
tests without rewriting the task or camera interfaces. If measured CAD changes
the agreed envelope, version that specification explicitly; do not silently
change success meaning. No CAD provenance is asserted before the files arrive.

## Side effects

```text
robot_connections = 0
robot_commands = 0
serial_connections = 0
physical_camera_connections = 0
openai_requests = 0
```

Only offline modules, fixtures, stored assets and MuJoCo rendering were used.
The tests audit forbidden hardware/provider imports and runtime object-pose
assignments. No robot/serial/camera device or Astra/OpenAI was accessed. Local
commits only; no push or automatic merge, and no root component-pin change.
