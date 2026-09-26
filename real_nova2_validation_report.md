# Nova2 Real Kinematic Validation

Status: **kinematic validation passed for the sampled configurations**.

This report is based on the read-only Dashboard dataset
`real_nova2_samples.json`. The capture utility sent only `RobotMode()`,
`GetErrorID()`, `GetAngle()`, and `GetPose(user=0,tool=0)`. No motion or state
changing command was sent by the capture utility.

## Dataset

- Samples: 12
- Distinct joint vectors: 10 (`pose_01`--`pose_03` repeat one pose)
- User frame: 0
- Tool frame: 0
- Joint units: degrees
- TCP pose position units: millimetres
- Observed `GetPose` orientation convention: XYZ RPY, degrees
- Raw dataset SHA-256: `3687a2a901132497c9e50b426004ba3ffa9b3b52860ec04b43ad29d62f0e280c`

The original dataset remains unchanged. Its metadata was provisional
`rotation_vector`; the comparison used the explicit `--orientation rpy`
override after checking the physical samples.

## Comparison definition

For each q vector, MuJoCo FK was evaluated for both hypotheses:

- `current`: official pinned description, local `+Z` axes
- `legacy`: `q_legacy = [-q1, q2, q3, -q4, -q5, q6]`

Position error is `p_sim - p_real` in millimetres. Orientation uses matrices
and the relative rotation:

`R_err = R_real.T @ R_sim`

The orientation error is the angle of `R_err`; Rx/Ry/Rz components are never
subtracted directly.

## Results

| Axis hypothesis | Position mean | Position median | Position RMS | Position max | Orientation mean | Orientation median | Orientation RMS | Orientation max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 0.010808 mm | 0.010645 mm | 0.010882 mm | 0.012292 mm | 0.000471 deg | 0.000472 deg | 0.000472 deg | 0.000538 deg |
| legacy | 361.876422 mm | 286.138904 mm | 444.942427 mm | 1058.967634 mm | 140.851389 deg | 168.856395 deg | 148.010835 deg | 177.487150 deg |

The current hypothesis has approximately constant millimetre-scale residuals
with mean offset `[-0.006, 0.003, 0.001] mm`; no TCP/base/tool-frame offset is
indicated. The legacy residual is configuration-dependent and is rejected by
the data.

## Reproduction

```bash
cd /home/dsa/project/dobot/nova_mujoco
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python \
  compare_pose.py real_nova2_samples.json --orientation rpy \
  | tee real_nova2_validation_rpy.txt
```

The standalone MJCF was not modified as part of this validation. Dynamics,
collision fidelity, load behavior, and RL/imitation-learning behavior remain
out of scope.
