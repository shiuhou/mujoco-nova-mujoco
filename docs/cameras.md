# Offline camera and manipulation validation

`nova_cameras.py` snapshots the complete simulator state into a render-only model,
producing front_rgb, right_rgb and wrist_rgb from one state generation. Render
failures cannot reuse a previous observation. Source dynamics and arm geometry
remain unchanged. Camera/renderer parameter provenance is in `configs/`.

The release passes 213 offline tests with MuJoCo 2.3.7 and EGL llvmpipe (Mesa 25.2.8,
LLVM 20.1.2). On the reviewed host, EGL device 0 is AMD radeonsi and device 1 is
llvmpipe. **Device indices are host-specific**: inspect the renderer metadata on
another machine and rerun validation; do not silently substitute a backend.

On that host the explicit commands are:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python -m pytest -q
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python run_s4_camera_validation.py --output evidence/camera/<new-run>
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python run_s3_s4_integration.py --output evidence/integration/<new-run>
```

Outputs use new directories and record renderer/runtime metadata. These commands
use no physical camera, serial device or model endpoint. The Python audit guard is
an additional test backstop, not an OS sandbox.

AMD radeonsi on this host fails the unchanged static-repeat gate: one observed
front-view comparison differed at 7 pixels with maximum channel difference 13,
while state/hash/poses were identical. llvmpipe produced exact pixels for the same
comparison and passes the suite. This isolates a backend-dependent discrepancy;
it does not prove the low-level driver cause is fixed. Software rendering can be
slower and no throughput improvement is claimed. Pixel tolerances have not been
widened, and no cached/quantized replacement images are used.

The S5 model loop is still published on research/astra-mujoco-embodiment. Its CLI
requires historical integration evidence and, for S5B, a prior live S5A PASS and Git
ancestry. Removing those checks would discard an existing acceptance requirement.
A future promotion must define a portable, source-bound prerequisite contract and
retain live-provider acceptance; this release neither copies raw history nor skips
those checks. No model API calls were made during this publication review.
