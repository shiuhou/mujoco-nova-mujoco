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

## Roadmap

- **S3 — gripper and object interaction:** attach the extracted official finger
  geometry only after the flange transform is established; add the nominal
  95 mm opening, cube/box contact, grasp/release, and contact calibration.
- **S4 — camera parity:** provide `front_rgb`, `right_rgb`, and `wrist_rgb`
  with documented approximate poses, fields of view, and resolutions.
- **S5 — GPT-6 Astra closed-loop simulation:** route the generic agent
  `move_to` and `ActionChunk` through the simulator and feed fresh MuJoCo
  observations back to the next decision.
- **S6 — sim versus real experiment:** hold task, policy, initial state, and
  action semantics constant while comparing action sequences, success, effect,
  oscillation, collision diagnostics, grasp outcomes, and camera changes.

These are documentation milestones only. S1/S2 does not call Astra, OpenAI,
the live agent provider, a robot, a serial device, or a physical camera.
