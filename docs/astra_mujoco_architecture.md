# Astra / Robocurve to Nova2 MuJoCo architecture

This phase establishes the simulator-side execution boundary for the future
Astra comparison. The policy-facing representation remains the generic
Robocurve EEF action; the embodiment chooses whether that action is executed by
the real Dobot adapter or by this offline MuJoCo adapter.

```text
Astra / policy
      |
      v
move_to -> ActionChunk -> execution adapter
                              |             |
                       real Dobot       NovaMujocoSimulator
                       MovL boundary    IK -> bounded q target
                                             -> kinematic execution
```

`NovaMujocoChunkExecutor` is a thin public-data protocol boundary. It accepts
an iterable of seven-value actions or an object exposing `.actions`; action
objects may expose `.data`. It does not import private Robocurve code and it
does not change `inspect-robots-dobot`.

The simulator has one required S1 mode: `kinematic`. A command is a bounded
linear joint reference trajectory. The fixed simulation timestep is 0.002 s,
and the default engineering speed bound is 1.0 rad/s. These values are
`SIMULATION_ENGINEERING_PARAMETERS`; they are not measured Nova controller
parameters. No wall-clock sleep is used.

The controlled pose is explicit. `link6` is the validated flange/tool frame.
`nominal_grasp_center` is an optional nominal geometry site with a +Z 0.197 m
offset; it is not a calibrated TCP. The Cartesian action is expressed in the
MuJoCo base frame and targets the selected controlled frame.

S1/S2 expose joint state, controlled pose, simulation time, and approximate
collision contact diagnostics. S4 now adds `NovaMujocoCameraRig`: it composes
three configured cameras into a separate render model and snapshots the source
state without changing the simulator. `observe()` returns coherent native RGB
for `front_rgb`, `right_rgb`, and `wrist_rgb`, plus the corresponding base-relative
`eef_state`. The fixed external views and rigid Link6 wrist view have explicit
engineering transforms. See [S4 camera parity](sim_tracks/s4_camera_parity.md)
for real-source evidence, timing semantics and the future Observation mapping.
S3 adds `ManipulationSimulator` in explicit
`kinematic_arm_with_dynamic_objects` mode and `ManipulationChunkExecutor`.
It owns the benchmark objects, nominal gripper, contact physics and task metrics.
The integrated camera rig expands this modular production MJCF into a separate
render model and reports measured gripper aperture in `eef_state`. Camera-only
test fixtures are not part of production. See
[S3×S4 integration](sim_tracks/s3_s4_integration.md) for the full observed
manipulation proof and the production-only engineering wrist transform.
Astra, OpenAI requests, real hardware and calibrated dynamics remain outside
this boundary; no S5 execution is introduced.

The source model and FK convention remain unchanged from the validated Nova2
model. The existing current-axis real-sample comparison is the geometry
baseline; the execution layer must not be interpreted as a dynamics or
sim-to-real validation.
