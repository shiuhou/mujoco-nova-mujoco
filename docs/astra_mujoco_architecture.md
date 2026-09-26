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
collision contact diagnostics. Future observations can attach renderers to the
logical names `front_rgb`, `right_rgb`, and `wrist_rgb`; no cameras are created
in this phase. Astra, OpenAI requests, real hardware, gripper contact, and
dynamic calibration are outside this boundary.

The source model and FK convention remain unchanged from the validated Nova2
model. The existing current-axis real-sample comparison is the geometry
baseline; the execution layer must not be interpreted as a dynamics or
sim-to-real validation.
