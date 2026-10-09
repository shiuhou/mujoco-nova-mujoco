# S4 three-camera observation evidence

Status: **FAIL**

Code revision: `4d199b10bca2a3e1cccb9e6c6f2997c97a1aeba2`

Native-resolution uint8 RGB; test fixtures only. No physical calibration claim.

| View | States/s (FPS for one view) | Mean ms/state |
|---|---:|---:|
| front_rgb | 2646.97 | 0.38 |
| right_rgb | 2696.17 | 0.37 |
| wrist_rgb | 1577.86 | 0.63 |
| all_three | 555.60 | 1.80 |

All camera images use one copied simulation state; static/reset pixels repeat exactly on this stack.
Fixed cameras remain fixed; wrist follows Link6. Parameters and full residuals are in JSON.

No robot, serial, physical camera, or OpenAI access.
