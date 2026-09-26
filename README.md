# Dobot Nova2 MuJoCo Phase 1

This directory contains the offline simulation and model foundation for the
integrated Astra-to-Nova research workspace. It currently provides a MuJoCo
model converted from the official Dobot ROS2 V4 Nova2 description, FK/pose
comparison tools, and a reproducible place for future tool, gripper,
simulation, and sim-to-real experiments.

The directory is intentionally hardware-independent by default. Offline model
results support research and regression checks; they do not grant live robot
permission or prove collision, dynamics, payload, or TCP safety. The nested
`DOBOT_6Axis_ROS2_V4` checkout remains an upstream/reference boundary rather
than an automatically connected runtime.

## Source

- Repository: `DOBOT_6Axis_ROS2_V4`
- Revision: `0f21e9839888f188ac06b0a9fea622d13a29a74f`
- Kinematics: `DOBOT_6Axis_ROS2_V4/cra_description/urdf/nova2_robot.xacro`
- Meshes: `DOBOT_6Axis_ROS2_V4/cra_description/meshes/nova2/*.STL`

The model preserves the official visual meshes. Collision geometry is simple
box proxies derived from the STL bounds. Dynamics are not tuned and there are
no actuators, RL components, or hardware connections.

The seven official STL files are copied into `assets/` so `nova2.xml` remains
loadable without a ROS installation or the source repository. `ASSET_LICENSE`
contains the upstream license.

See [`axis_conventions.md`](axis_conventions.md) for the complete link/joint
audit, frame convention, and current versus legacy axis hypothesis.

## Official X-Trainer gripper asset

The official Dobot X-Trainer Hugging Face dataset publishes the complete USD
robot asset. It does not publish the original SolidWorks `SLDASM` or editable
CAD features, but its USD assembly contains the finger meshes visible on the
physical tool:

```text
Dobot-robot/X-trainer
  x-trainer/assets/robots/X-Trainer-URDF/urdf/
    x-trainer_asm-0226.SLDASM/
      configuration/x-trainer_asm-0226.SLDASM_base.usd
```

The pinned dataset revision used here is
`02f008252001f654e28dee40d615c713ff79935a`. The downloaded source files are
under `assets/xtrainer_official/`. The base USD is 45,127,540 bytes and uses
metres with Z up. The source stage hash and all extracted mesh hashes are in
`assets/xtrainer_gripper/manifest.json`.

The extracted official finger prims are:

```text
/visuals/J1_7/J1_7/mesh
/visuals/J1_8/J1_8/mesh
/visuals/J2_7/J2_7/mesh
/visuals/J2_8/J2_8/mesh
```

Each has 15,276 USD vertices and 5,092 triangles. Its local extent is about
48.0 x 122.9 x 26.0 mm. Both STL and OBJ exports are provided. Recreate them
from the pinned USD with:

```bash
python -m pip install usd-core
python extract_xtrainer_gripper.py \
  --part J1_7 --part J1_8 --part J2_7 --part J2_8
```

The exports are deliberately in each USD mesh prim's local frame. The manifest
records the X-Trainer assembly transform, but that transform is not a Nova2
`Link6` or flange transform. A measured flange-to-tool transform is still
required before these meshes can be attached to `nova2.xml` or used as a
gripper TCP.

The legacy `x-trainer` `DobotRobot` code selects controller `Tool(1)` with
`SetTool(1, 0, 0, 197, 0, 0, 0)`. That is a controller frame setting rather
than gripper geometry or proof of the physical fingertip TCP. The legacy
gripper class only maps servo positions.

The extraction validation recorded for this asset set is **17 tests passed**;
Ruff check and format checks passed. USD parsing used `usd-core 26.8`. No
robot, camera, or serial device was accessed. These checks validate offline
loading and export integrity only; they do not validate a Nova2 flange mount,
tool frame, collision fit, or physical TCP.

## Offline FK

Install the MuJoCo Python binding in the active environment:

```bash
python -m pip install mujoco
```

Then run:

```bash
python nova_fk.py --q-deg 0 0 0 0 0 0
python nova_fk.py --q-rad 0 0.4684 -1.3719 0 1.7695 0
python nova_fk.py --q-deg 10 20 30 40 50 60 --axis-hypothesis legacy
```

The script prints the TCP position in metres and millimetres, a quaternion,
and a mathematical rotation vector in degrees. `GetPose` Rx/Ry/Rz values on
the observed controller use XYZ RPY instead. Its main position is relative to
`base_link`; it also prints the raw world-frame position. It never connects to
a robot.

## Interactive visualization

From a terminal with a working X11/Wayland `DISPLAY`, run:

```bash
./run_visualize.sh --q-deg 0 0 0 0 0 0
```

The launcher first uses the active `python`, then falls back to the existing
MuJoCo environment at `/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/`.
You can also invoke that environment explicitly:

```bash
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python visualize.py --q-deg 0 0 0 0 0 0
```

Or, after installing MuJoCo into the active environment:

```bash
python -m pip install mujoco
python visualize.py --q-deg 0 0 0 0 0 0
```

Other useful poses:

```bash
python visualize.py --q-deg 10 20 30 40 50 60
```

The window displays the official STL visual geometry. Drag to orbit, scroll to
zoom, and close the window to exit. To inspect the historical axis hypothesis:

```bash
python visualize.py --q-deg 10 20 30 40 50 60 --axis-hypothesis legacy
```

Collision boxes are hidden by default so they do not cover the official mesh.
Use `--show-collision` when checking the simple Phase 1 collision proxies.

This viewer is offline and does not connect to or command a robot.

## Phase 1.5 read-only real-sample capture

`capture_real_pose.py` is the only utility in this directory intended to
connect to the Nova Dashboard. It is strictly read-only and sends only these
four commands:

```text
RobotMode()
GetErrorID()
GetAngle()
GetPose(user=0,tool=0)
```

It never requests control, powers or enables the robot, clears alarms, or
sends motion commands. It displays the paired readings, requires an explicit
`SAVE` confirmation for every sample, and appends to the output file unless
`--overwrite` is explicitly supplied. The capture metadata records the
observed `rpy` convention for Dobot Rx/Ry/Rz. Each sample uses a short
Dashboard connection and closes it before waiting for confirmation, so the
normal teleoperation process can use port 29999 between captures.

## Real Nova2 kinematic validation procedure

When next to the physical Nova2:

1. Ensure the robot is stationary and the work area is safe.
2. In one terminal, run:

   ```bash
   cd /home/dsa/project/dobot/nova_mujoco
   python capture_real_pose.py --output real_nova2_samples.json
   ```

3. Review the displayed mode, errors, joint angles, and pose. Type `SAVE` to
   store `pose_01` (or enter a sample name).
4. At the `Capture another sample? [Y/n]` prompt, leave the capture terminal
   waiting. Its Dashboard connection is already closed.
5. Use the existing approved safe teleoperation stack in a separate terminal
   to move to another stable pose. The capture utility itself must remain
   read-only and must not move the robot.
6. Return to the capture terminal and press Enter for `Y`. It opens one short
   read-only connection, reads the next paired state, and closes it again.
7. Review the values and save the next sample. Repeat until 5–10 diverse safe
   poses are recorded.
8. Prefer different shoulder configurations, clearly bent and extended elbow
   configurations, different J4/J5/J6 combinations, and different workspace
   directions. Avoid obvious singular configurations during the first pass.
9. Run the offline comparison. For the original 12-sample file captured with
   provisional `rotation_vector` metadata, include the explicit RPY override:

   ```bash
   /home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python \
     compare_pose.py real_nova2_samples.json --orientation rpy
   ```

The comparison evaluates both the current and legacy axis hypotheses for every
sample. It prints real and simulated TCP positions, `dx/dy/dz`, position error
norm, orientation error angle, aggregate mean/median/RMS/max statistics, and
diagnostic patterns. It uses `R_err = R_real.T @ R_sim`; it does not subtract
Euler or rotation-vector components.

New datasets created by the updated capture utility already declare `rpy`, so
the override can be omitted for them.

The resulting file is real ground-truth evidence only after the operator has
collected it next to the robot. Synthetic fixtures under `fixtures/` are for
offline tests and must not be presented as physical validation.

## Collision proxy review

The current collision model uses one box per official visual mesh. These boxes
are geometry proxies, not tuned contact shapes. Audit them against the pinned
STL assets with:

```bash
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python \
  collision_review.py
```

The audit checks each proxy in its link-local frame. All seven current boxes
enclose their corresponding STL AABBs within the numerical tolerance, with
proxy-to-mesh AABB volume ratios within about 0.0004% of one. This confirms no
visual geometry is omitted from a proxy, but an AABB can still create false
contacts around concave or hollow regions. Collision fidelity therefore remains
open; do not treat this audit as a dynamics or contact-model validation.

Use `visualize.py --show-collision` for a visual inspection of the boxes. The
kinematic MJCF and its validated axis conventions remain unchanged.

## Offline GetAngle/GetPose comparison

Create a record from a read-only observation. The following values are only an
illustration of the input format, not a robot observation:

```json
{
  "user": 0,
  "tool": 0,
  "q_deg": [0, 0, 0, 0, 0, 0],
  "pose_mm_deg": [300, -100, 200, 180, 0, 90],
  "orientation": "rpy"
}
```

The legacy single-record format remains supported. Run:

```bash
python compare_pose.py recorded_pose.json
python compare_pose.py recorded_pose.json --axis-hypothesis legacy
```

The comparison reports position error in millimetres and orientation error in
degrees for both the current and legacy axis hypotheses by default. The record
must explicitly specify `user=0, tool=0`; other frames require measured frame
transforms and are rejected. The record must also specify its orientation
convention. The model TCP is the `Link6` origin.

The first read-only physical dataset contains 12 samples at 10 distinct joint
vectors; `pose_01` through `pose_03` repeat the same configuration. With its
recorded Rx/Ry/Rz interpreted as XYZ RPY, the current-axis model gives
0.0109 mm position RMS (0.0123 mm maximum) and 0.00047 degree orientation
RMS (0.00054 degree maximum). The legacy-axis model gives 444.9 mm position
RMS and 148.0 degree orientation RMS. This validates the current model's
kinematic agreement with `GetPose(user=0,tool=0)` over the sampled poses;
it does not establish collision or dynamic accuracy. The original dataset
is preserved with its provisional orientation metadata.

See [`real_nova2_validation_report.md`](real_nova2_validation_report.md) for
the reproducible command, dataset hash, and complete acceptance summary. The
raw per-sample terminal output is preserved in
`real_nova2_validation_rpy.txt`.

## Regression checks

```bash
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m pytest -q --ignore=DOBOT_6Axis_ROS2_V4
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m ruff check --exclude DOBOT_6Axis_ROS2_V4 .
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python -m ruff format --check --exclude DOBOT_6Axis_ROS2_V4 .
```

`capture_real_pose.py` is read-only when intentionally run beside the robot;
all other scripts are offline. No script in this directory powers, enables,
clears, or moves a physical robot.

## Asset hashes

```text
Link1.STL     dfa2d03d4d5ef76a16874af3db8303b8ab642182a22d27856b0d16515b710c79
Link2.STL     83f56d3e22493104861db5d3ccc75f95f60f74a16b730381a1e6e58bd8fbb6c2
Link3.STL     a5d536e9807eaee98bee06b504d42e305389b9284494ffbe970ad333b6db2c86
Link4.STL     c62ef326c5f9b7cc2ae314e4592084a9fd5d312f2783cf28a36ea2f52dae0178
Link5.STL     fff9c1170d17d237b157ae2f2ea8fdb05febaedbe9072872b9d4f3e83659dcc8
Link6.STL     872a1c4fa13b45749a482e8c26c2d9144f1f6d8d49dd1109e10c6345f19d197c
base_link.STL 944a15e41eccba51aada79332f66703b641c9e1f46ecdd5fe9f0f2e775f0cc2d
```
