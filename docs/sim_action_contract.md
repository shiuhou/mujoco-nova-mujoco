# Simulation action contract

This table preserves the S1/S2 action checkpoint. S4 now implements the logical
image boundary; see [S4 observations](sim_tracks/s4_camera_parity.md) for exact
RGB shapes, reference evidence, camera calibration gaps and freshness semantics.

The contract below was checked against the read-only `inspect-robots-dobot`
source and its pinned upstream references. The simulator consumes the same
generic fields; it does not define an Astra-specific action language.

| Field / concept | Real inspect-robots-dobot meaning | S2 MuJoCo meaning | Status |
|---|---|---|---|
| `x`, `y`, `z` | Absolute EEF target in metres, base/user frame 0 | Absolute target in metres, MuJoCo base frame | MATCH |
| `yaw`, `pitch`, `roll` | Reset-relative radians with `Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ R_reference` | Same reference-orientation contract; S2 accepts only zero values | MATCH for pinned zero subset; arbitrary orientation DEFERRED |
| `gripper` | Zero-pinned inactive `NoOpGripper` placeholder | Must be exactly zero; no finger state or contact command | DEFERRED / inactive |
| action shape | `[x,y,z,yaw,pitch,roll,gripper]`, seven values | Same seven-value public shape | MATCH |
| action frame | `base`, corresponding to Dobot `user=0` in the current profile | `base`, relative to the MuJoCo `base_link` pose | MATCH for current profile |
| controlled tool frame | Explicit tool/flange frame selected by the embodiment | `link6` site by default; nominal grasp center is separately selectable | MATCH for Link6/flange; grasp-center calibration DEFERRED |
| reference orientation | Measured at reset; zero deltas preserve it | Rotation matrix captured at simulator reset and reused by every S2 target | MATCH |
| waypoint progression | `move_to` produces absolute linear waypoints and an `ActionChunk`; pre-check sees the full proposal | Each waypoint is solved and settled in order; first failure rejects the remainder | MATCH in the tested execution seam |
| observations | `eef_state` is position plus reset-relative orientation and inactive gripper; cameras are named by embodiment | Clean robot state and logical camera-name hooks; no image renderer yet | State MATCH in scope; images DEFERRED |

`inspect-robots-agent` linearly interpolates named absolute coordinates and marks
only the final action with `chunk_final`. The simulator preserves waypoint
ordering and records every requested action, IK result, achieved pose, and
simulation time. It does not treat `chunk_final` metadata as authority and it
does not import private framework internals.

Position values are SI metres and orientation slots are radians at this
boundary. The real Dobot TCP command later converts to native millimetres and
degrees; that conversion is not performed by the simulator adapter.

The +Z 10 mm regression and the six-direction benchmark exercise the pinned
translation contract with orientation held at the reset reference. They do not
establish native Dobot orientation interpolation, physical path clearance, or
controller equivalence.
