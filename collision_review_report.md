# Nova2 Collision Proxy Review

This is a geometry audit of the Phase 1 collision boxes. It does not validate
contact behavior, friction, dynamics, or self-collision behavior.

Command:

```bash
/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python collision_review.py
```

Tolerance: `2e-6 m`.

| Link | Official STL | Triangles | Encloses STL AABB | Max boundary error | Proxy/AABB volume |
|---|---|---:|---|---:|---:|
| base_link | base_link.STL | 6162 | yes | 0.000378 mm | 1.000004 |
| Link1 | Link1.STL | 9408 | yes | 0.000062 mm | 1.000000 |
| Link2 | Link2.STL | 11568 | yes | 0.000210 mm | 0.999999 |
| Link3 | Link3.STL | 11604 | yes | 0.000061 mm | 1.000000 |
| Link4 | Link4.STL | 13950 | yes | 0.000077 mm | 1.000000 |
| Link5 | Link5.STL | 14550 | yes | 0.000195 mm | 1.000004 |
| Link6 | Link6.STL | 4514 | yes | 0.000063 mm | 0.999998 |

All seven boxes are the expected bounds-derived AABBs. They conservatively
cover the official mesh extents, but may occupy empty space around concave or
hollow geometry and therefore produce false contacts. No MJCF dimensions were
changed during this review.
