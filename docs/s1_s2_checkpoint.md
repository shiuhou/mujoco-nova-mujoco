# S1/S2 checkpoint

Status: **S1 PASS / S2 PASS for the offline action-interface scope.**

The checkpoint evidence is the immutable directory
[`evidence/sim_s1_s2/20260927T-s1-s2-checkpoint-002`](../evidence/sim_s1_s2/20260927T-s1-s2-checkpoint-002/),
generated from simulator revision `af26bff235224fbda1314bd7139ffe2e198943ef`.
The earlier directory is retained and marked superseded because its per-case
`action_frame` label was wrong; its numerical results were unaffected.

## Existing model

The source is the pinned official `DOBOT_6Axis_ROS2_V4` Nova2 description,
converted to `nova2.xml` with the validated current local +Z joint-axis
convention. The controlled baseline is the Link6 origin (`tcp` site), expressed
relative to `base_link`. Existing FK validation remains unchanged:

| Metric | Current-axis result |
|---|---:|
| Position RMS | 0.010882 mm |
| Position maximum | 0.012292 mm |
| Orientation RMS | 0.000472 deg |
| Orientation maximum | 0.000538 deg |

## S1 result

The deterministic `kinematic` reference mode loads the model unchanged,
supports reset/readback, derives limits from the MJCF, rejects invalid or
out-of-range q targets, and advances a fixed 0.002 s simulation clock. Its
linear qpos reference uses a 1.0 rad/s engineering speed bound. There is no
calibrated position-servo or Nova dynamics claim.

## S2 result

The bridge uses MuJoCo site Jacobians and bounded damped least squares. The
controlled frame is `link6`; action coordinates are absolute `base` positions
in metres. Orientation is held at the reset reference, and gripper is inactive.
The default position/orientation tolerances are `1e-6 m` and `1e-6 rad`, with
200 maximum iterations and explicit failure statuses. Eight validated starts ×
six 10 mm directions produced 48/48 successful solves.

Aggregate benchmark results:

| Measure | Result |
|---|---:|
| Position residual RMS / max | 0.000197614 / 0.000534850 mm |
| Orientation residual RMS / max | 0.000011159 / 0.000039141 deg |
| IK iterations mean / max | 2.3125 / 4 |
| Joint displacement mean / max | 0.082772 / 0.203274 rad |
| IK throughput | 21.1 solves/host s |
| MuJoCo forward throughput | 252,526 steps/host s |
| Contact-reporting cases | 4 / 48 |

Contact reports are coarse Link4/Link6 box-proxy diagnostics at `pose_06`; they
do not establish a collision-safe path.

Per-direction results across the eight starts:

| Direction | Pass | RMS error (mm) | Max error (mm) | Mean iterations | Contact cases |
|---|---:|---:|---:|---:|---:|
| +X | 8/8 | 0.000231400 | 0.000534850 | 2.250 | 1 |
| -X | 8/8 | 0.000228805 | 0.000530394 | 2.250 | 0 |
| +Y | 8/8 | 0.000225949 | 0.000441311 | 2.625 | 1 |
| -Y | 8/8 | 0.000162715 | 0.000393810 | 2.500 | 0 |
| +Z | 8/8 | 0.000160860 | 0.000311601 | 2.125 | 1 |
| -Z | 8/8 | 0.000158134 | 0.000339684 | 2.125 | 1 |

## +Z 10 mm parity regression

The representative `pose_01` case starts at
`[-0.014481, -0.640625, 2.157494, 0.025498, -1.548077, -1.556851]` rad with
Link6 position `[-0.181492072, -0.116882686, 0.375477036]` m. The requested
target is the same x/y, +0.010 m z, and the reset rotation. The achieved pose is
`[-0.181492096, -0.116882685, 0.385476950]` m, giving
`dx=-2.44e-8 m`, `dy=3.92e-10 m`, `dz=0.009999914 m`, position residual
`8.94097e-8 m` (`0.00008941 mm`), and orientation residual
`3.83599e-12 rad` (`2.20e-10 deg`).

This is an action-semantics regression, not dynamics validation.

## Quality gates and side effects

The final local checks are 68 pytest tests, Ruff check, Ruff format check,
Python compilation, `git diff --check`, and the existing real-sample FK
comparison command. The nested ROS2 checkout is excluded from test discovery as
an upstream reference boundary. No hardware or model provider was accessed:

- `robot_connections = 0`
- `robot_commands = 0`
- `serial_connections = 0`
- `physical_camera_connections = 0`
- `openai_requests = 0`

The simulator is ready for **S3 design work** and has the state hooks needed for
future **S4 camera parity**, but neither phase is started by this checkpoint.
