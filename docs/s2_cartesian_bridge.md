# S2 Cartesian bridge

`CartesianIKSolver` uses MuJoCo's site Jacobian for the selected controlled
frame and a deterministic damped-least-squares update. It starts from the
current validated joint state unless a validated seed is supplied, limits each
joint update to 0.08 rad, and stops after at most 200 iterations. Joint limits
are checked on every proposed update. The default tolerances are:

- position: `1e-6 m` (`0.001 mm`)
- orientation: `1e-6 rad`
- damping: `1e-4`
- stagnation window: 20 iterations

The solver reports `converged`, `joint_limit`, `stagnated`, or
`max_iterations`. Only `converged` is accepted by the executor. A failed or
invalid target is an explicit rejection; the closest pose is never reported as
success. Targets must be finite, have a proper rotation matrix, and remain
within the 0.02 m maximum Cartesian step used by the chunk executor.

`NovaMujocoChunkExecutor` captures the reset reference orientation and maps
each action to a pose with that orientation. It executes every waypoint in
order, solves IK, commands the resulting joint target, settles it, records
contacts and residuals, and stops at the first rejected waypoint. S2 currently
requires yaw, pitch, roll, and gripper to be zero. This is deliberate: it
validates translation and reference-orientation semantics before introducing
arbitrary orientation changes or gripper behavior.

The controlled frame is written into every benchmark record. Cartesian action
coordinates are in `base`; the default controlled frame is `link6`. The
nominal grasp-center site can be selected explicitly and is based on the
operator-provided +Z 197 mm nominal geometry, not a precision TCP calibration.

The offline benchmark covers eight recorded, validated starting configurations
and all six 10 mm directions. It reports IK status, iterations, joint
displacement, achieved pose, residuals, simulation time, and coarse contact
diagnostics in JSON. See the immutable evidence record linked from the
checkpoint document.
