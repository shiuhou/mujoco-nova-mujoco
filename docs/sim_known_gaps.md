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
finger joints and object contact, calibrated TCP/tool geometry, physical camera
pose/FOV calibration, Astra/OpenAI closed-loop calls, and any sim-to-real behavioral
comparison. The model is a kinematically validated Nova2 simulator, not an
unqualified 1:1 digital twin.

**S4 observation parity implemented:** three exact logical RGB names, Phase 5A
resolutions, deterministic offscreen rendering, fixed front/right and Link6
wrist attachment, coherent image/state snapshots and increasing frame generation.
All camera poses/FOVs remain explicit engineering assumptions. Physical
intrinsics, distortion, hand-eye, exposure timing, room appearance and noise are
unknown/unmodeled. The table/blue/black/red scene is a non-contact visual fixture,
not S3 production objects. See [S4 contract](sim_tracks/s4_camera_parity.md).

## Roadmap

- **S3 — gripper and object interaction:** attach the extracted official finger
  geometry only after the flange transform is established; add the nominal
  95 mm opening, cube/box contact, grasp/release, and contact calibration.
- **S4 — camera parity:** logical observation implementation delivered on the
  isolated S4 branch; physical calibration and timing matching remain open.
- **S5 — GPT-6 Astra closed-loop simulation:** route the generic agent
  `move_to` and `ActionChunk` through the simulator and feed fresh MuJoCo
  observations back to the next decision.
- **S6 — sim versus real experiment:** hold task, policy, initial state, and
  action semantics constant while comparing action sequences, success, effect,
  oscillation, collision diagnostics, grasp outcomes, and camera changes.

S3/S5/S6 remain documentation milestones in this branch. S4 calls no Astra,
OpenAI, live agent provider, robot, serial device, or physical camera.
