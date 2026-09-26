# S1 deterministic execution layer

`NovaMujocoSimulator` is the S1 execution primitive. It loads `nova2.xml`,
reads all six joint limits directly from the MJCF, and exposes:

```python
reset(q=None)
get_joint_position()
get_link6_pose()
get_tool_pose(frame=...)
command_joint_target(q_target)
step(steps=...)
settle()
contact_diagnostics()
close()
```

Reset validates a six-element finite vector, sets qpos and zero qvel, calls
MuJoCo forward kinematics, resets simulation time to zero, and captures the
reference rotation. Joint targets outside the model ranges are rejected. NaN,
infinity, wrong dimensions, and malformed values are rejected; impossible
targets are never silently clamped.

The required control mode is a kinematic reference mode. A target is linearly
interpolated in qpos at a fixed 0.002 s timestep, with a default maximum joint
speed of 1.0 rad/s. `step()` advances simulation time deterministically and
never waits for host wall time. `settle()` advances until the pending reference
trajectory is complete. The model has no calibrated position-servo or Nova
motor dynamics in S1, so no position-servo mode is advertised.

The six model ranges currently are:

| Joint | Lower (rad) | Upper (rad) |
|---|---:|---:|
| joint1 | -6.28 | 6.28 |
| joint2 | -3.14 | 3.14 |
| joint3 | -2.79 | 2.79 |
| joint4 | -6.28 | 6.28 |
| joint5 | -6.28 | 6.28 |
| joint6 | -6.28 | 6.28 |

These are model limits, not a safety certification for a physical robot.
Contact reports name the coarse collision proxy geoms and penetration distance;
they are research diagnostics only.
