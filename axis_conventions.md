# Nova2 Axis Conventions

## Source contract

The source is the official `DOBOT_6Axis_ROS2_V4` repository, pinned locally at
commit `0f21e9839888f188ac06b0a9fea622d13a29a74f`. The Nova2 description is
`cra_description/urdf/nova2_robot.xacro`.

URDF uses metres and radians. The MuJoCo model uses the same units. Dobot
Dashboard `GetAngle` uses degrees, and `GetPose` uses millimetres plus degrees.

## Link and joint frames

The fixed chain is:

`world -> dummy_link -> base_link -> Link1 -> Link2 -> Link3 -> Link4 -> Link5 -> Link6`

The `world_joint` places `dummy_link` at `(0, 0, 0.03)`. `dummy_joint` is
identity. The six revolute joints are represented in MuJoCo by child bodies
with the URDF joint origin translation and rotation. The body rotations are
stored as quaternions so the URDF fixed-axis RPY convention is explicit:

`R = Rz(yaw) * Ry(pitch) * Rx(roll)`

The MuJoCo hinge `axis="0 0 1"` is a vector in the corresponding child body
frame, which is the same local joint frame used by each URDF `<axis>` element.

| Joint | URDF origin xyz | URDF origin rpy | Local axis in source | Zero-pose axis in parent frame |
|---|---|---|---|---|
| joint1 | `(0, 0, 0.2234)` | `(0, 0, 0)` | `+Z` | `+Z` |
| joint2 | `(0, 0, 0)` | `(1.5708, 1.5708, 0)` | `+Z` | `-Y` |
| joint3 | `(-0.28, 0, 0)` | `(0, 0, 0)` | `+Z` | `+Z` |
| joint4 | `(-0.22501, 0, 0.1175)` | `(0, 0, -1.5708)` | `+Z` | `+Z` |
| joint5 | `(0, -0.12, 0)` | `(1.5708, 0, 0)` | `+Z` | `-Y` |
| joint6 | `(0, 0.088004, 0)` | `(-1.5708, 0, 0)` | `+Z` | `+Y` |

The official file currently uses `+Z` for all six local axes. Git history shows
an earlier Nova2 revision with `-Z` for J1, J4, and J5. This is the known
inconsistency and is not silently resolved. `nova_fk.py --axis-hypothesis
legacy` evaluates that historical hypothesis by negating q1, q4, and q5
before using the current MJCF.

## TCP and pose comparison

The official description has no separate TCP or tool link. The model therefore
defines the `tcp` site at the origin of `Link6`. This corresponds to a
zero-offset tool frame only; it does not claim to reproduce a calibrated tool.

The MJCF preserves the source `world_joint` offset for visual compatibility.
`nova_fk.py` reports both the raw world-frame site position and a pose relative
to `base_link`. The latter is used for `GetPose(user=0, tool=0)` comparison.

The comparison interface expects an offline record with `user=0`, `tool=0`,
`q_deg` (the matching `GetAngle` vector), and `pose_real_mm_deg` (the matching
`GetPose` vector). The legacy single-record field `pose_mm_deg` remains
supported. The record must declare its orientation convention. The current
and legacy axis hypotheses are both reported by default.

## Phase 1.5 orientation comparison

The multi-sample capture format stores `pose_real_mm_deg`. The initial
rotation-vector interpretation was tested against 12 read-only Nova2 samples
and rejected: it produced 38--179 degree errors. Interpreting the same fields
as XYZ roll/pitch/yaw (`orientation: "rpy"`) produced 0.00040--0.00054 degree
errors. New captures therefore declare `orientation: "rpy"`. This is an
empirical result for the observed controller/API behavior, not a component-wise
Euler subtraction.

Orientation is compared through matrices. For each sample and axis hypothesis,
the relative rotation is defined in this direction:

`R_err = R_real.T @ R_sim`

The reported orientation error is the angle of `R_err` in degrees. Rx/Ry/Rz
components are never subtracted directly.

The read-only capture utility records `RobotMode`, `GetErrorID`, `GetAngle`,
and `GetPose(user=0,tool=0)` only. It does not require mode 5 to read state,
but refuses to save a normal sample for active motion, initialization,
collision, alarm, unknown, or non-empty error states.
`compare_pose.py --orientation rpy` can be used for older datasets captured before this
convention was updated.
