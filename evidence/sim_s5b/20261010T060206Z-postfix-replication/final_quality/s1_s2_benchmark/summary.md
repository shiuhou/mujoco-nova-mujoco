# Nova2 MuJoCo S1/S2 Offline Benchmark

Generated: `2026-10-10T06:10:44.860327+00:00`
Simulator revision: `5d10695b866eab79619bb7a3cf8f73e5103b95d2`

This is a simulator action-interface characterization. It is not a real Nova
controller-accuracy, dynamics, collision-safety, or hardware validation claim.

## Aggregate

- Cases: 48/48 PASS (100.0%)
- Position residual RMS/max: 0.000198 / 0.000535 mm
- Orientation residual RMS/max: 0.000011159 / 0.000039141 deg
- IK iterations mean/max: 2.312 / 4
- Joint displacement mean/max: 0.082772 / 0.203274 rad
- Cases reporting coarse-proxy contacts: 4
- IK solves per host second: 20.9
- MuJoCo forward steps per host second: 252265.3

## Per direction

| Direction | Pass | RMS error (mm) | Max error (mm) | Mean iterations | Contacts |
|---|---:|---:|---:|---:|---:|
| +X | 8/8 | 0.000231 | 0.000535 | 2.250 | 1 |
| -X | 8/8 | 0.000229 | 0.000530 | 2.250 | 0 |
| +Y | 8/8 | 0.000226 | 0.000441 | 2.625 | 1 |
| -Y | 8/8 | 0.000163 | 0.000394 | 2.500 | 0 |
| +Z | 8/8 | 0.000161 | 0.000312 | 2.125 | 1 |
| -Z | 8/8 | 0.000158 | 0.000340 | 2.125 | 1 |

The JSON artifact contains initial and achieved joint states and poses, requested
and actual deltas, IK status/iterations/residuals, simulation time, and named geom
contact diagnostics for every case.

## Side effects

- robot_connections = 0
- robot_commands = 0
- serial_connections = 0
- physical_camera_connections = 0
- openai_requests = 0
