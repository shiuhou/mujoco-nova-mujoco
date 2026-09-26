# X-Trainer End-Effector Extraction

The source USD uses generic assembly component names rather than semantic
names such as `gripper`, `palm`, or `flange`. The geometry audit gives this
working mapping:

| USD component | Local extent (mm) | Evidence | Status |
| --- | ---: | --- | --- |
| `J1_7`, `J1_8` | 48.0 x 122.9 x 26.0 | mirrored pair at the distal end | finger candidates |
| `J2_7`, `J2_8` | 48.0 x 122.9 x 26.0 | mirrored duplicate pair for the second X-Trainer side | finger candidates |
| `J1_6`, `J2_6` | 160.0 x 136.0 x 122.5 | larger components immediately before the paired fingers | end-effector body candidates |

The four `J*_7`/`J*_8` meshes are the components that match the photographed
tapered white finger geometry. The two `J*_6` meshes are included in the
official USD stage but are not extracted here because they may be part of the
X-Trainer wrist/actuator housing rather than the printable finger adapter.
The USD does not identify a separate Nova2 flange adapter or a TCP marker.

The extracted STL and OBJ files remain in the USD mesh prim-local frame. The
assembly translation and rotation for each part are recorded in
`manifest.json`; they are X-Trainer assembly transforms and must not be copied
to the Nova2 `Link6` frame. A physical flange-to-tool calibration is required
before using any of these meshes as a Nova2 tool or TCP.

The `J1` and `J2` groups are two copies of the X-Trainer's end-effector
assembly. They are not Nova2 joint names and do not describe Nova2 kinematics.
