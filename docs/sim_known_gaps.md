# Known gaps and later phases

## Current fidelity boundaries

**Controller agreement only:** nominal Nova2 FK convention against the recorded
controller sample set, not measured physical geometry. The current-axis result is 0.010882 mm position RMS,
0.012292 mm maximum, 0.000472 degree orientation RMS, and 0.000538 degree
maximum over the existing samples.

**Implemented:** deterministic qpos reference execution, explicit fixed
timestep, model-derived joint-limit rejection, Link6 and nominal grasp-center
state, bounded Cartesian IK, reset-relative orientation preservation, ordered
waypoint execution, residual logging, and approximate contact diagnostics.

**Approximate:** kinematic controller engineering parameters and coarse box
collision proxies. Contacts are not collision-safe path evidence; four cases in
the benchmark reported Link4/Link6 proxy contacts at `pose_06`.

**Deferred:** calibrated motor/controller dynamics, payload and friction,
physical collision/contact fidelity, native ServoJ/MovL equivalence, gripper
physical gripper/contact fidelity, calibrated TCP/tool geometry, camera pose/FOV
matching, Astra/OpenAI closed-loop calls, and any sim-to-real behavioral
comparison. The model is a kinematically validated Nova2 simulator, not an
unqualified 1:1 digital twin.

## Current scope and next work

- Contact manipulation is published: nominal 95 mm gripper, contact grasp/lift/
  transport/release and settled containment. See [usage](manipulation.md).
- Three-camera/contact integration is published and validated on explicit EGL llvmpipe.
  AMD EGL repeatability remains unresolved on this host.
- Astra closed-loop simulation remains on
  [the research branch](https://github.com/shiuhou/mujoco-nova-mujoco/tree/32efb430d17c23e28e068b7751afda48509ab885).
  One model-driven placement succeeded; historical acceptance evidence/ancestry
  dependencies still block portable S5 CLI promotion.
- Physical camera/TCP/contact calibration and sim-to-real comparison remain open.
  The official finger meshes require a measured flange transform before adoption.

The main manipulation runner is offline and uses no model provider or physical
robot/camera/serial device. Simulation success is not physical validation.


## Current renderer and promotion boundary

The camera/contact integration is now available on main with 213 passing tests on
explicit EGL llvmpipe. AMD EGL repeatability still fails on the reviewed host; see
[camera validation](cameras.md). Historical evidence/ancestry dependencies still
block S5 CLI promotion, and prior model-run success is not a portable main result.
