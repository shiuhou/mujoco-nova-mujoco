# Known gaps and later phases

## Current fidelity boundaries

**Validated:** official Nova2 geometry and FK convention against the recorded
real sample set. The current-axis result is 0.010882 mm position RMS,
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
finger joints and object contact, calibrated TCP/tool geometry, camera pose/FOV
matching, Astra/OpenAI closed-loop calls, and any sim-to-real behavioral
comparison. The model is a kinematically validated Nova2 simulator, not an
unqualified 1:1 digital twin.

## Current scope and next work

- Contact manipulation is published: nominal 95 mm gripper, contact grasp/lift/
  transport/release and settled containment. See [usage](manipulation.md).
- Three-camera observations and Astra closed-loop simulation are available on
  [the research branch](https://github.com/shiuhou/nova-mujoco/tree/32efb430d17c23e28e068b7751afda48509ab885).
  One model-driven placement succeeded. Current renderer repeatability fails
  revalidation; CLI historical-evidence and ancestry dependencies also block a
  standalone main release. These layers remain research, not published main runtime.
- Physical camera/TCP/contact calibration and sim-to-real comparison remain open.
  The official finger meshes require a measured flange transform before adoption.

The main manipulation runner is offline and uses no model provider or physical
robot/camera/serial device. Simulation success is not physical validation.
