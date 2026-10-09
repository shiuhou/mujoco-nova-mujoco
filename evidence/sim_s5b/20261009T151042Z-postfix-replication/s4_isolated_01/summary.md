# S4 three-camera observation evidence

Status: **PASS**

Code revision: `4d199b10bca2a3e1cccb9e6c6f2997c97a1aeba2`

Native-resolution uint8 RGB; test fixtures only. No physical calibration claim.

| View | States/s (FPS for one view) | Mean ms/state |
|---|---:|---:|
| front_rgb | 2606.81 | 0.38 |
| right_rgb | 2575.73 | 0.39 |
| wrist_rgb | 1636.26 | 0.61 |
| all_three | 495.94 | 2.02 |

All camera images use one copied simulation state; static/reset pixels repeat exactly on this stack.
Fixed cameras remain fixed; wrist follows Link6. Parameters and full residuals are in JSON.

No robot, serial, physical camera, or OpenAI access.
