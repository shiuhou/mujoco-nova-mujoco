# S4 three-camera observation evidence

Status: **PASS**

Code revision: `093e79c555fa22d9b23b997e1d0a6d5687bdae08`

Native-resolution uint8 RGB; test fixtures only. No physical calibration claim.

| View | States/s (FPS for one view) | Mean ms/state |
|---|---:|---:|
| front_rgb | 54.75 | 18.26 |
| right_rgb | 53.28 | 18.77 |
| wrist_rgb | 70.71 | 14.14 |
| all_three | 19.27 | 51.90 |

All camera images use one copied simulation state; static/reset pixels repeat exactly on this stack.
Fixed cameras remain fixed; wrist follows Link6. Parameters and full residuals are in JSON.

No robot, serial, physical camera, or OpenAI access.
