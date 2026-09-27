# Agent Instructions

This repository is the offline Nova2 MuJoCo, FK, tool-geometry and future
sim-to-real research foundation. Read `README.md` and the relevant file under
`docs/` before changing model, action-contract or evidence files.

## Boundaries

- Keep the default simulator, FK tools and tests hardware-independent.
- Offline simulation results are regression evidence only. They do not prove
  controller accuracy, collision safety, payload limits, TCP calibration or
  physical robot safety.
- Do not connect to, enable, power, clear errors on, or move a Nova while
  working in this repository unless the operator has separately authorized a
  specific live read-only or motion procedure.
- `capture_real_pose.py` is the documented read-only exception and still
  requires current authorization.
- Treat `DOBOT_6Axis_ROS2_V4/` as an upstream/reference boundary. Do not edit
  or vendor its contents as part of simulator changes.
- Do not infer flange, tool, user-frame, gripper polarity or TCP transforms
  from meshes or photographs. Record measured transforms and uncertainty.
- Preserve official asset provenance, revisions, licenses and hashes when
  adding or regenerating meshes.

## Validation

Use the repository's offline tests and benchmark before a checkpoint:

```bash
python -m pytest
python run_s1_s2_benchmark.py --output evidence/sim_s1_s2/<new-unique-name>
```

Never overwrite existing evidence directories. Keep new evidence separate and
state whether any hardware, camera, serial or network resource was accessed.

For Dobot TCP/IP facts, consult the workspace primary reference
`/home/dsa/project/dobot/dobot_tcp_ip_reference.md` and the original PDF when
the extracted Markdown is ambiguous. Do not invent undocumented protocol facts.

The workspace-level coordination rules are in the parent repository's
`AGENTS.md`, `README.md`, `HANDOFF.md` and `components.lock.yaml`.
