# Nova2 FK, IK and calibration audit

Audit date: 2026-10-27. **Dream-1 physical action execution is not ready.** All
work here is offline: no robot, camera, serial device, controller settings or
provider API was accessed. Model-generated actions remain disconnected.

## Scope and classifications

Baseline: simulator `c7534fc` (its FK/IK implementation is identical to the
older local component `d811bcd`); X-Trainer `42358b9`. Original implementations
remain reproducible through Git. Work uses separate worktrees so unrelated
component and root edits are preserved. No arm geometry or production IK method
was replaced. The SciPy optimizer below is audit-only.

| Component | Classification | What the classification covers |
| --- | --- | --- |
| Nominal FK chain | VERIFIED WITH LIMITATIONS | Broad URDF/MJCF mathematical consistency; shared nominal geometry |
| Recorded controller agreement | VERIFIED WITH LIMITATIONS | GetAngle/GetPose, 12 samples / 10 distinct configurations; not physical accuracy |
| Cartesian Jacobian | VERIFIED WITH LIMITATIONS | Rotated-base defect fixed; finite-difference regression; fixed-base model only |
| Local DLS IK | VERIFIED WITH LIMITATIONS | Good nearby-seed results; poor global/limit coverage; not a collision planner |
| Global multi-branch/nearest-solution IK guarantee | MISSING | Single local seed does not enumerate or rank all solutions |
| TCP pivot fitter | VERIFIED WITH LIMITATIONS | Synthetic recovery and independent SVD agree; frame provenance defect fixed |
| Physical pinch-center TCP and orientation | IMPLEMENTED BUT UNVALIDATED | Nominal geometry and inconsistent pivot data; no independent acceptance |
| Gripper polarity record | INCORRECT / NEEDS FIX | Conflicting historical and newer operator records; no value guessed here |
| Nova2 camera intrinsics/distortion, hand-eye, front-to-base | MISSING | No measured record or Nova2 calibration pipeline found in inspected sources |
| Legacy RGB/depth registration | INCORRECT / NEEDS FIX | Reader returns separate unaligned streams; not valid registered RGB-D geometry |
| Physical camera/TCP accuracy and timing | MISSING | No held-out metrology or exposure synchronization evidence |

The classifications distinguish mathematical consistency, controller agreement,
and independently measured physical accuracy. None of the latter is established.

## FK review

`nova_fk.tcp_pose` sets measured-order J1..J6 in MuJoCo, calls `mj_forward`,
and expresses the `tcp` site in `base_link` using the inverse base rotation and
translation. It validates six finite radians but deliberately does not enforce
joint limits (the simulator does). The standalone FK reloads MJCF every call;
that is suitable for the CLI, not a high-rate loop. The simulator caches its model.

The source is vendor `DOBOT_6Axis_ROS2_V4` revision
`0f21e9839888f188ac06b0a9fea622d13a29a74f`,
`cra_description/urdf/nova2_robot.xacro`. The six numeric origin transforms,
local +Z axes, J1..J6 ordering, limits and zero joint references match the MJCF.
The complete origin table is in [axis_conventions.md](../../axis_conventions.md).
No additional encoder-zero correction or calibrated DH parameters are present.
URDF RPY is fixed-axis XYZ: `Rz(yaw) Ry(pitch) Rx(roll)`; lengths/radians are SI.
The source uses rounded 1.5708 rad and MJCF rounded quaternions. World mounting
adds +30 mm Z; base-relative FK removes it. `tcp` is **Link6 origin**, not a
measured pinch center. `nominal_grasp_center` adds local +197 mm Z; the supplied
meshes and nominal site do not establish its physical accuracy. J1/J2 limits
are ±3.14, J3 ±2.79, J4..J6 ±6.28 rad, inherited model limits rather than verified
current controller/cable limits.

The independent reference parses numeric vendor joint XML and multiplies
homogeneous transforms with SciPy rotations, without using MuJoCo FK or its
Jacobian. It evaluates 353 configurations: 256 uniform joint-space samples,
64 near-limit corners, 32 elbow/wrist singular cases, plus zero. This is broad
joint-space coverage, not uniform Cartesian volume or collision-free coverage.

| Mathematical FK metric | Result |
| --- | --- |
| Position RMS / maximum vs independent evaluator | 0.00000335 / 0.00000672 mm |
| Rotation RMS / maximum | 1.365e-8 / 1.863e-8 rad |

These tiny differences fit transform serialization precision. This independently
checks the implementation, **not the source geometry**: both evaluators share
vendor link dimensions and zero/axis assumptions.

### Re-examining the 12-sample result

Re-running the preserved dataset with the explicit `rpy` override reproduces
0.0108818 mm position RMS / 0.0122919 mm maximum and 0.000472206° rotation RMS /
0.000537910° maximum. Three entries repeat one configuration. The alternative
legacy axis signs give 444.942 mm and 148.011° RMS. The initially recorded
rotation-vector metadata was rejected and RPY selected using this same data;
it is not an independent held-out orientation-convention validation.

GetAngle and GetPose come from the same controller; the controller reports its
own kinematic estimate. The vendor URDF and firmware can share nominal geometry.
The exact firmware calibration implementation is unavailable. Sequential reads
also are not hardware-synchronized. This comparison checks signs, units and
conventions on the sampled controller, not flex, backlash, mounting, payload,
absolute physical accuracy, TCP fit, or all-workspace behavior. Four-decimal
joint/pose serialization limits residual interpretation. Do not fit away these
small discrepancies or claim 11-micrometre physical precision.

No recorded PositiveKin/InverseKin reference set was found. The TCP/IP reference
specifies degree joints and mm/degree poses, explicit user/tool indices, and
InverseKin `useJointNear=1,jointNear={...}`. No live calls were made. A future
separately authorized query-only comparison can use the same sampled q/targets,
record exact firmware/frame/seed/error metadata and compare residuals. It would
still be controller agreement, not external metrology.

## IK review and quantitative results

`CartesianIKSolver` is single-seed numerical damped least squares, not analytical
IK. It stacks translational/rotational MuJoCo site Jacobians, uses damping 1e-4,
limits each joint update to 0.08 rad, and allows 200 iterations. Position and
rotation tolerances are separately 1e-6 m and 1e-6 rad. It restores q after the
query. It immediately rejects an update crossing a limit; it does not project,
line-search, restart or enumerate branches. Mixed metre/radian residual weighting
and stagnation scoring are numerical choices, not a calibrated task metric.
The condition number is similarly scale-dependent and is not a safety threshold.

Fixed seed 20261027; nearby seed noise sigma=0.05 rad clipped to limits. Identical
sampling before/after the frame fix gives identical success counts:

| Reachable target group | Nearby seed | Zero seed |
| --- | --- | --- |
| Uniform joint-space targets | 256/256 (100%) | 63/256 (24.6%) |
| All near-limit corners | 3/64 (4.7%) | 0/64 |
| Singular/near-singular targets | 32/32 (100%) | 8/32 (25%) |

The failures are explicit joint_limit, stagnated or max_iterations, not accepted
approximate solutions. All returned joint vectors remain within model limits.
Near-limit failures occur even for known reachable FK targets: immediate rejection
of a boundary-crossing Newton update is not proof of unreachable workspace.
Zero is itself singular, so zero-seed figures are deliberately difficult, not a
forecast for every live use. Nearby success does not establish global robustness.

Corrected-run latency (one host, single-thread OpenBLAS, all successes and failures):

| Group | Median / P95 / P99 milliseconds |
| --- | --- |
| Random, nearby seed | 0.308 / 0.432 / 0.595 |
| Random, zero seed | 11.410 / 12.351 / 12.823 |
| Limits, nearby seed | 0.139 / 0.237 / 2.941 |
| Limits, zero seed | 2.840 / 2.971 / 3.452 |
| Singular, nearby seed | 0.518 / 3.652 / 7.527 |
| Singular, zero seed | 11.133 / 12.295 / 12.412 |

Timing is host/load/version dependent, not a real-time guarantee. Successful
random-near solutions have position median/P95/P99/max
0.00000175/0.000380/0.000796/0.000995 mm and orientation
1.71e-9/4.56e-7/8.20e-7/8.84e-7 rad. Full success-only and all-attempt residual
and latency distributions are in generated JSON; failed residuals are not hidden.

Two deterministic 101-waypoint reachable trajectories (ordinary and wrist-zero
crossing) both complete. Initial seed acquisition is about 0.1 rad; subsequent
adjacent-step statistics are recorded separately by the current runner. This is
about 0.0020003 rad maximum on the ordinary path and 0.0020013 rad on the
singularity-crossing path. This is
sampling consistency only, with no velocity/acceleration, collision or branch
continuity guarantee. Among 63 random targets solved from both seeds, all 63
returned joint vectors differed by >0.1 rad (maximum L2 13.61 rad, including
periodic winding differences). The solver biases toward its seed but does not
prove the globally nearest configuration; naive wrap-to-pi could violate bounds
or cable winding and was not added.

The independent reference uses SciPy bounded `least_squares`, finite-difference
Jacobians and the parsed URDF FK, with zero seed and 200 function evaluations.
For the same 32 targets it achieves 23/32 versus production 5/32. Median/P95/P99
latency is 9.719/17.828/35.062 ms versus 11.517/12.034/12.651 ms. Both results are
accepted only by separate pose residuals, not optimizer exit flags. Evaluation
budgets differ from IK iterations. SciPy improves this global subset but has
slower tails, still fails 9 poses, adds a dependency and has not established
branch continuity; it remains a separate benchmark, not a replacement solver.

### Confirmed defects and fixes

1. `site_jacobian` returned **world** derivatives while `get_tool_pose` and IK
   residuals use **base** coordinates. Identity mounting hid the mismatch.
   Rotating the base by XYZ [0.4,-0.6,1.2] rad and requesting +X 5 mm previously
   stagnated with 41.779 mm and 0.007415 rad residual. Rotating both Jacobian
   blocks into base coordinates now converges to 0.0000130 mm and 5.13e-8 rad.
   Independent central differences check every translational/rotational column.
2. IK iteration counts accepted fractional, NaN and infinite values, producing
   late exceptions or invalid stagnation bounds. Both counts now require positive
   integers. Regressions failed before the fix and pass afterward.
3. X-Trainer's TCP loader discarded user/tool provenance. An already offset
   Tool 1 pose could be fitted and falsely labelled flange-to-TCP. It now requires
   explicit base/flange metadata and rejects conflicting per-sample frames;
   collection rejects nonzero frames before connecting. Bare historical lists
   require provenance review, not automatic relabelling.

The original local IK algorithm, limits and geometry remain unchanged apart from
the frame correction and parameter validation. No physical action path was added.

## TCP/gripper evidence

`tcp_calibration.solve_pivot` fits `p_i + R_i t = c` for flange-relative translation
`t` and fixed base point `c`, using normal equations and pivoted elimination.
It estimates **no tool orientation**. Normal equations square conditioning;
rank-poor samples must not be rescued by an arbitrary precision claim. The new
independent diagnostic uses SVD and leave-one-out residuals without changing the
production fit or accepting a Tool definition.

The existing private eight-pose file was transcribed from terminal output after
save failure and has no capture timestamps. Recalculation gives candidate
`t=[-0.203,-2.032,202.246] mm`, fit RMS **7.0034 mm**, max **10.3193 mm**.
Design rank=6, condition=2.7432; SVD differs by only 2.2e-13 mm. Thus this dataset's
large residual is not explained by numerical ill-conditioning. Contact changes,
point definition, readings or transcription need investigation; the root physical
cause cannot be established from these records. Leave-one-out errors range
3.52–12.32 mm. First-six fit / last-two exploratory split gives 5.69 and 10.71 mm
held-out pivot inconsistency, **not independent 3D accuracy**.

X-Trainer YAML records nominal [0,0,197] mm and zero rotation as operator-confirmed,
with 5 mm measurement uncertainty. Simulator calibration YAML leaves mount
transforms null and physically_verified=false. Those are different evidence
levels, not a completed calibration. Official USD mesh coordinates are not Nova
flange coordinates. The grasp center can vary with jaw opening, finger contact
and load. Historical YAML says open=1470/closed=2490; newer workspace notes report
the reverse for the installed mechanism. Do not pick a polarity without renewed
measured evidence. These existing records were preserved, with the contradiction
made explicit; no physical calibration values or controller settings changed.

## Camera evidence and weaknesses

Inspected inspect camera/Frame/V4L2 code, Dream observation adapter, X-Trainer
RealSense reader, local simulator `s4_cameras.json` / `s3_s4_cameras.json`, and
Argon Nova task mapping. No measured Nova2 calibration artifact with K, distortion,
resolution, target detections, calibrated transforms and held-out residuals was
found in the inspected workspace records. Broader unrelated upstream examples
and Franka intrinsics are not evidence for this rig.

- Front/wrist RGB shape, uint8, acquisition/publication times and age/skew are
  checked. These are transport validity checks, not calibration or exposure sync.
- Simulator views use configured camera placement/FOV. Wrist placement is marked
  production-only with no measured hand-eye transform. Argon axis-token mapping
  explicitly says projection/hand-eye are unknown; image-relative labels are not
  a base-camera rotation matrix.
- Legacy RealSense code starts 640x480 color/depth at 90 Hz, optionally resizes and
  rotates both arrays. It does not call `rs.align`, export sensor intrinsics or
  depth scale, retain hardware timestamps, or update K after image transforms.
  Equal image dimensions do not make depth registered to color. Bilinear resizing
  of depth also mixes discontinuities. This is a raw preview interface, not a
  calibrated 3D sensing pipeline; it was not imported or connected in this audit.
- No Nova2 reprojection RMS, per-view errors, hand-eye observability diagnostics,
  surveyed 3D validation, exposure skew or depth-registration residual is available.
  All those metrics are **NOT_MEASURED**, not zero. Low synthetic rendering error
  or low 2D fit residual would not establish accurate base-coordinate targets.

## Physical acceptance and supervised measurement procedure

The intended object clearance, task accuracy, load, approved workspace, speed,
and instrument uncertainty are unspecified. Therefore physical thresholds remain
**UNSET**, and execution is blocked. Numerical regression thresholds (1e-6 m/rad
for IK, 1e-8 derivative tolerance, 1e-8 m / 3e-8 rad URDF serialization agreement)
are software criteria only. Do not reuse them as achievable physical accuracy.

Before a supervised measurement session, define smallest grasp/placement clearance
`C`, required success percentile, validated speed `v`, tool lever arm `L`, and
instrument uncertainty. Allocate error budgets so conservative contributions from
base/FK, TCP, camera/extrinsics, tracking and timing satisfy
`E_base + E_TCP + E_camera + E_tracking + v*delta_t + L*delta_theta < C/2`
with an explicitly chosen margin. Use measured worst-case or confidence bounds;
RSS combination requires justified independence. Require held-out P95/max errors
plus measurement uncertainty below each allocated budget. At nominal L=197 mm,
1° angular uncertainty alone contributes about 3.44 mm lateral error.

No commands below authorize hardware access or movement. A separate reviewed
session must specify operator, robot identity/firmware, tool/user settings,
payload, safe poses, limits, emergency stop and current permission.

1. **External FK validation:** use a calibrated tracker/CMM or surveyed fiducial
   rig capable of measuring both position and orientation; record its uncertainty.
   Define base and flange datums independently, avoid fitting and testing the same
   poses, sample approved diverse poses/load/approach directions, repeat after
   remounting. Preserve raw encoder, controller pose and independent instrument
   readings with timestamps. Report translation and geodesic rotation errors,
   bias, repeatability, P95/max and uncertainty separately; no controller pose
   may substitute for the independent measurement.
2. **TCP:** install a rigid traceable pointer/reference nest, or a fixture locating
   the actual pinch center at a defined opening. Use calipers/gauge blocks for
   finger geometry and gap; physical point repeatability must beat the allocated
   error budget. Collect diverse rotations about multiple axes with Tool0/User0,
   stationary confirmation, errors/mode and timestamps; translation-only or
   single-axis orientation sets are inadequate. Preassign fit and held-out sets.
   Fit offline, inspect SVD/rank/residuals, then test held-out pivots and separate
   surveyed locations. Estimate tool orientation from independently observed
   non-collinear datums; pivot fitting alone cannot recover it. Repeat for relevant
   openings/load/remounts. Review the candidate before any separately authorized
   controller write. Reconfirm servo endpoints/polarity and measured gap under
   separate gripper authorization.
3. **Intrinsics:** use a dimensionally verified flat checkerboard/ChArUco target
   and recorded raw images, with focus/exposure/resolution/crop/flip fixed. Capture
   full-image coverage, tilts and distances; inspect corners and per-view
   residuals, reserve images and physical distances for validation. Record K,
   distortion model/coefficients, pixel convention, resolution, device/config
   identity privately, target dimensions/uncertainty and raw-data hashes. Treat
   resizing/rotation as an explicit image-coordinate transform of calibration.
4. **Extrinsics:** define `T_A_B` as mapping B coordinates into A. For wrist use
   `T_base_target = T_base_flange * T_flange_camera * T_camera_target`; fit a rigid
   camera mount using diverse nonparallel robot rotations and stationary pairs.
   OpenCV `calibrateHandEye` expects gripper-to-base and target-to-camera and
   returns camera-to-gripper; verify directions with synthetic known transforms
   before fitting physical data. For front camera, independently survey target
   poses in base and solve `T_base_camera = T_base_target * inverse(T_camera_target)`
   across views. Validate both cameras on held-out surveyed 3D points through the
   same production transform chain; report 3D bias/P95/max, not just pixel RMS.
5. **Timing/depth:** record sensor exposure and host timestamps, estimate clock
   offset/drift, and measure multi-camera/robot alignment (hardware trigger or
   observable LED/event plus independent timing). Validate at intended speed
   against `v*delta_t` budget. If depth is used, retrieve the actual factory
   intrinsics/extrinsics/scale, align depth to RGB explicitly and test surveyed
   targets at multiple depths and image locations. Preserve raw depth semantics;
   validate occlusions, invalid pixels and any resampling separately.

Only after these gates, compatible Dream checkpoint/action semantics, reviewed
joint/velocity/acceleration/collision limits and existing motion-gate/broker tests
may a separate task consider supervised action-chunk execution.

## Reproduction and validation

Use Python 3.11, numpy 1.26.4, mujoco 2.3.7 and scipy 1.15.3 for comparable numeric
results; pytest/Ruff are test tools. SciPy is needed only for the audit/reference,
not normal simulator use. Existing MuJoCo environment was used; Dream's API env
was not modified. In the simulator repository:

```bash
python -m pip install -r requirements-audit.txt
python -m pytest -q
python -m ruff check --exclude DOBOT_6Axis_ROS2_V4 .
python -m ruff format --check --exclude DOBOT_6Axis_ROS2_V4 .
OPENBLAS_NUM_THREADS=1 python kinematics_audit.py \
  --urdf /path/to/pinned/DOBOT_6Axis_ROS2_V4/cra_description/urdf/nova2_robot.xacro \
  --count 256 --output evidence/kinematics/new-run
python run_s1_s2_benchmark.py --output evidence/kinematics/new-s1s2
python compare_pose.py real_nova2_samples.json --orientation rpy
```

The vendor checkout is a read-only external reference, not copied into this repo.
Use exactly the revision above; the output records its file hash, model/code
hashes, seed and dependency versions. A new output directory is mandatory.
Baseline source is preserved by Git `c7534fc`. To compare without replacing the
current implementation, create an isolated baseline checkout, then invoke the new
runner with that checkout as the import root:

```bash
git worktree add --detach /path/to/baseline-checkout c7534fc
cd /path/to/baseline-checkout
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -c \
  'import runpy; runpy.run_path("/path/to/current-checkout/kinematics_audit.py", run_name="__main__")' \
  --urdf /path/to/pinned/DOBOT_6Axis_ROS2_V4/cra_description/urdf/nova2_robot.xacro \
  --count 256 --output evidence/kinematics/baseline-comparison
```

The runner hashes its own file and the selected implementation separately. Reports store derived synthetic
rows, not local endpoints or raw hardware traces. The local before/fixed/final
runs live under `evidence/kinematics/`; generated evidence remains ignored.

In X-Trainer, install its existing `requirements-teach-dev.txt`, then:

```bash
PYTHONPATH=source/leisaac:source/leisaac/leisaac/xtrainer_utils python -m pytest -q
python tcp_calibration_audit.py /path/to/private/pivot-samples.json --output /path/to/new-pivot-report.json
```

The pivot input remains private. Its recorded SHA256 is
`dda48417f1e74fa0639a4b0d76114dbc18cc56f86054652fb39e53d375cc0f34`.
Reprojection and independent 3D error cannot be reproduced until measured camera
records exist. That absence is a deliverable limitation, not a simulated substitute.

Validation actually run: simulator **140 PASS**, existing S1/S2 benchmark
**48/48**, full simulator Ruff lint/format **PASS**; X-Trainer **83 PASS**, changed
calibration modules mypy and Ruff lint/format **PASS**. The new frame/iteration
regressions first produced 8 failures; TCP provenance regressions first produced
4 failures. Both sets pass after the fixes. Initial X-Trainer full-suite execution
in the MuJoCo environment failed on missing pyserial; rerunning in a dedicated
worktree environment installed from its existing dev requirements passed. No
hardware result is inferred from these tests. Physical metrics remain NOT_MEASURED.
