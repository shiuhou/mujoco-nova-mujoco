# S4 three-camera observation evidence

Status: **PASS**

Code revision: `4d199b10bca2a3e1cccb9e6c6f2997c97a1aeba2`

Native-resolution uint8 RGB; test fixtures only. No physical calibration claim.

| View | States/s (FPS for one view) | Mean ms/state |
|---|---:|---:|
| front_rgb | 2599.96 | 0.38 |
| right_rgb | 2609.88 | 0.38 |
| wrist_rgb | 1576.66 | 0.63 |
| all_three | 642.37 | 1.56 |

All camera images use one copied simulation state; static/reset pixels repeat exactly on this stack.
Fixed cameras remain fixed; wrist follows Link6. Parameters and full residuals are in JSON.

No robot, serial, physical camera, or OpenAI access.
