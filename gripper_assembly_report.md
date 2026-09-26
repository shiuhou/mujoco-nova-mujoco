# X-Trainer End-Effector Assembly Audit

Source: `/home/dsa/project/dobot/nova_mujoco/assets/xtrainer_official/X-Trainer-URDF/urdf/x-trainer_asm-0226.SLDASM/configuration/x-trainer_asm-0226.SLDASM_physics.usd`

## Scope and frame boundary

The composed `physics.usd` stage is audited in its X-Trainer assembly/world frame. Mesh prims are stage-level `/visuals/.../mesh` assets; component bodies are flat children under the assembly root. No prim is named `NovaFlange`, `TCP`, or `gripper`, so a Nova flange transform is not asserted from names alone.

## Hierarchy (actual prim paths)

```text
/x_trainer_asm_0226_SLDASM
  /base_link
  /J1_1 -> /J1_2 -> /J1_3 -> /J1_4 -> /J1_5 -> /J1_6
      /J1_6 -> /J1_7 (prismatic finger) and /J1_8 (prismatic finger)
  /J2_1 -> /J2_2 -> /J2_3 -> /J2_4 -> /J2_5 -> /J2_6
      /J2_6 -> /J2_7 (prismatic finger) and /J2_8 (prismatic finger)
  /joints/J1_1 ... /J4_6 (physics joints)
```

The arrows above are joint `body0 -> body1` relationships, not USD parent nesting: all component bodies are direct children of the assembly root.

## Component transforms and meshes

| Prim | Parent | local translate (m) | world translate (m) | visual mesh | APIs |
|---|---|---:|---:|---|---|
| `/x_trainer_asm_0226_SLDASM/base_link` | `/x_trainer_asm_0226_SLDASM` | `(0, 0, 0)` | `[0.0, 0.0, 0.0]` | `/visuals/base_link/base_link/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_1` | `/x_trainer_asm_0226_SLDASM` | `(0, 0, 0.22339999675750732)` | `[0.0, 0.0, 0.22339999675750732]` | `/visuals/J1_1/J1_1/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_2` | `/x_trainer_asm_0226_SLDASM` | `(0, 0, 0.22339999675750732)` | `[0.0, 0.0, 0.22339999675750732]` | `/visuals/J1_2/J1_2/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_3` | `/x_trainer_asm_0226_SLDASM` | `(7.275957614183426e-12, -0.000001043081283569336, 0.5034000277519226)` | `[7.275957614183426e-12, -1.043081283569336e-06, 0.5034000277519226]` | `/visuals/J1_3/J1_3/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_4` | `/x_trainer_asm_0226_SLDASM` | `(0.22500082850456238, -0.11749942600727081, 0.503400444984436)` | `[0.22500082850456238, -0.11749942600727081, 0.503400444984436]` | `/visuals/J1_4/J1_4/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_5` | `/x_trainer_asm_0226_SLDASM` | `(0.34500083327293396, -0.117499440908432, 0.5034008622169495)` | `[0.34500083327293396, -0.117499440908432, 0.5034008622169495]` | `/visuals/J1_5/J1_5/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_6` | `/x_trainer_asm_0226_SLDASM` | `(0.3450014591217041, -0.117499440908432, 0.4156578779220581)` | `[0.3450014591217041, -0.117499440908432, 0.4156578779220581]` | `/visuals/J1_6/J1_6/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_7` | `/x_trainer_asm_0226_SLDASM` | `(0.369006872177124, -0.054999448359012604, 0.29997795820236206)` | `[0.369006872177124, -0.054999448359012604, 0.29997795820236206]` | `/visuals/J1_7/J1_7/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J1_8` | `/x_trainer_asm_0226_SLDASM` | `(0.369006872177124, -0.1799994558095932, 0.29997795820236206)` | `[0.369006872177124, -0.1799994558095932, 0.29997795820236206]` | `/visuals/J1_8/J1_8/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_1` | `/x_trainer_asm_0226_SLDASM` | `(1.059999942779541, 0, 0.22339999675750732)` | `[1.059999942779541, 0.0, 0.22339999675750732]` | `/visuals/J2_1/J2_1/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_2` | `/x_trainer_asm_0226_SLDASM` | `(1.059999942779541, 0, 0.22339999675750732)` | `[1.059999942779541, 0.0, 0.22339999675750732]` | `/visuals/J2_2/J2_2/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_3` | `/x_trainer_asm_0226_SLDASM` | `(1.059999942779541, -0.000001043081283569336, 0.5034000277519226)` | `[1.059999942779541, -1.043081283569336e-06, 0.5034000277519226]` | `/visuals/J2_3/J2_3/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_4` | `/x_trainer_asm_0226_SLDASM` | `(0.8350008130073547, -0.11750266700983047, 0.5033996105194092)` | `[0.8350008130073547, -0.11750266700983047, 0.5033996105194092]` | `/visuals/J2_4/J2_4/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_5` | `/x_trainer_asm_0226_SLDASM` | `(0.7150008082389832, -0.11750353872776031, 0.5033987164497375)` | `[0.7150008082389832, -0.11750353872776031, 0.5033987164497375]` | `/visuals/J2_5/J2_5/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_6` | `/x_trainer_asm_0226_SLDASM` | `(0.7150017619132996, -0.1175025925040245, 0.4156557023525238)` | `[0.7150017619132996, -0.1175025925040245, 0.4156557023525238]` | `/visuals/J2_6/J2_6/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_7` | `/x_trainer_asm_0226_SLDASM` | `(0.738997757434845, -0.05500149726867676, 0.29997649788856506)` | `[0.738997757434845, -0.05500149726867676, 0.29997649788856506]` | `/visuals/J2_7/J2_7/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |
| `/x_trainer_asm_0226_SLDASM/J2_8` | `/x_trainer_asm_0226_SLDASM` | `(0.7389991283416748, -0.18000151216983795, 0.29997560381889343)` | `[0.7389991283416748, -0.18000151216983795, 0.29997560381889343]` | `/visuals/J2_8/J2_8/mesh` | `PhysicsRigidBodyAPI,PhysicsMassAPI` |

## Joint and articulation data

The physics layer contains no mimic relationship. J1/J2 joints 1-6 are revolute; J1/J2 joints 7-8 are prismatic. All reported values below are authored USD values.

| Joint prim | type | body0 -> body1 | axis | lower | upper | localPos0 | localPos1 | drive |
|---|---|---|---|---:|---:|---|---|---|
| `/x_trainer_asm_0226_SLDASM/joints/J1_1` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/base_link'] -> ['/x_trainer_asm_0226_SLDASM/J1_1']` | `Z` | -179.908737 | 179.908737 | `(0, 0, 0.2234)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_2` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J1_1'] -> ['/x_trainer_asm_0226_SLDASM/J1_2']` | `Z` | -179.908737 | 179.908737 | `(0, 0, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_3` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J1_2'] -> ['/x_trainer_asm_0226_SLDASM/J1_3']` | `Z` | -179.908737 | 179.908737 | `(0, -0.28, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_4` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J1_3'] -> ['/x_trainer_asm_0226_SLDASM/J1_4']` | `Z` | -179.908737 | 179.908737 | `(0, 0.225, 0.1175)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_5` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J1_4'] -> ['/x_trainer_asm_0226_SLDASM/J1_5']` | `Z` | -179.908737 | 179.908737 | `(0, -0.12, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_6` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J1_5'] -> ['/x_trainer_asm_0226_SLDASM/J1_6']` | `Z` | -179.908737 | 179.908737 | `(0, 0.087743, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_7` | `PhysicsPrismaticJoint` | `['/x_trainer_asm_0226_SLDASM/J1_6'] -> ['/x_trainer_asm_0226_SLDASM/J1_7']` | `Z` | -0.0399999991 | 0 | `(-0.0625, -0.024005, 0.11568)` | `(0, 0, 0)` | `linear` |
| `/x_trainer_asm_0226_SLDASM/joints/J1_8` | `PhysicsPrismaticJoint` | `['/x_trainer_asm_0226_SLDASM/J1_6'] -> ['/x_trainer_asm_0226_SLDASM/J1_8']` | `Z` | 0 | 0.0399999991 | `(0.0625, -0.024005, 0.11568)` | `(0, 0, 0)` | `linear` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_1` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/base_link'] -> ['/x_trainer_asm_0226_SLDASM/J2_1']` | `Z` | -179.908737 | 179.908737 | `(1.06, 0, 0.2234)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_2` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J2_1'] -> ['/x_trainer_asm_0226_SLDASM/J2_2']` | `Z` | -179.908737 | 179.908737 | `(0, 0, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_3` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J2_2'] -> ['/x_trainer_asm_0226_SLDASM/J2_3']` | `Z` | -179.908737 | 179.908737 | `(0, -0.28, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_4` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J2_3'] -> ['/x_trainer_asm_0226_SLDASM/J2_4']` | `Z` | -179.908737 | 179.908737 | `(0.225, 0, 0.1175)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_5` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J2_4'] -> ['/x_trainer_asm_0226_SLDASM/J2_5']` | `Z` | -179.908737 | 179.908737 | `(-0.12, 0, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_6` | `PhysicsRevoluteJoint` | `['/x_trainer_asm_0226_SLDASM/J2_5'] -> ['/x_trainer_asm_0226_SLDASM/J2_6']` | `Z` | -179.908737 | 179.908737 | `(-0.087743, 0, 0)` | `(0, 0, 0)` | `angular` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_7` | `PhysicsPrismaticJoint` | `['/x_trainer_asm_0226_SLDASM/J2_6'] -> ['/x_trainer_asm_0226_SLDASM/J2_7']` | `Z` | -0.0399999991 | 0 | `(0.0625, 0.023995, 0.11568)` | `(0, 0, 0)` | `linear` |
| `/x_trainer_asm_0226_SLDASM/joints/J2_8` | `PhysicsPrismaticJoint` | `['/x_trainer_asm_0226_SLDASM/J2_6'] -> ['/x_trainer_asm_0226_SLDASM/J2_8']` | `Z` | 0 | 0.0399999991 | `(-0.0625, 0.023995, 0.11568)` | `(0, 0, 0)` | `linear` |

## Assembly interpretation

- Directly sourced: J1_6/J1_7/J1_8 and J2_6/J2_7/J2_8 geometry, body transforms, joint axes, limits, body relationships, and linear drive metadata.
- The J1_6/J2_6 meshes are the larger distal body components and are extracted as body candidates; the USD provides no semantic proof that either is a Nova flange adapter.
- J1_7/J1_8 and J2_7/J2_8 are opposing prismatic finger pairs. Their axis is USD `Z`; J1_7/J2_7 limits are [-0.04, 0] m and J1_8/J2_8 limits are [0, 0.04] m.
- No mimic relationship, coupled actuator rule, or explicit nominal/open/closed state was found.
- `T_flange_gripper = inverse(T_world_flange) @ T_world_gripper` is **not computed**: the stage has no identifiable Nova wrist/flange prim. The X-Trainer world frame and Nova2 Link6 frame are unrelated until measured or sourced from a Nova-mounted assembly.
- The conditional flange integration gate was evaluated: even if physical adapter, bolt pattern, orientation, and assembly equivalence are later confirmed, the USD prerequisite is currently false because no Nova flange frame is authored. Therefore the nominal transform is not allowed yet.
- The X-Trainer controller configuration provides a nominal Tool 0 TCP seed `[0, 0, 197, 0, 0, 0]` (mm, degrees, Dobot pose order). This is a TCP offset seed for physical calibration; it is not evidence of a USD flange prim and is not applied to Nova2.

## Quantities requiring physical measurement

Nova flange-to-body origin/orientation, tool/TCP frame, actual finger zero/open state, mechanical coupling between the two fingers, and whether this X-Trainer end-effector is the same hardware revision as the Nova gripper.

## Provenance

This report is generated from the pinned official USD physics composition. It does not modify `nova2.xml`, apply X-Trainer world transforms to Nova2, or claim physical fit.

## Phase 2.2A composed preview

`build_nova2_gripper.py` generates `nova2_gripper.xml` from the frozen
`nova2.xml`. It preserves the extracted prim-local meshes and applies the
official relative transforms from J1_6 to each finger:

| finger | translation relative to J1_6 (m) | slide axis | encoded physical travel |
|---|---:|---|---:|
| J1_7 | `[-0.0625004, -0.0240054, 0.115680]` | local `Z` | authored `0..40 mm` |
| J1_8 | `[+0.0624996, -0.0240054, 0.115680]` | local `Z` | authored `0..40 mm` |

The authored USD limits are `[-40, 0] mm` and `[0, +40] mm`, while the
physical specification supplied for this tool is 95 mm maximum opening and
0 mm minimum gap. The generated preview keeps the authored 40 mm limits; it
does not impose the 95 mm opening as a mesh displacement because that would
contradict the official relative finger placement. The 95 mm and 0 mm values
remain physical calibration constraints in `gripper_calibration.yaml`, and
the 7.5 mm-per-side discrepancy is explicit. The grasp-center site is the
supplied nominal `[0, 0, 197] mm` from Link6.

This is a visual/kinematic preview only. It does not add servo mapping, mimic
coupling, contact tuning, or a physically verified adapter. The validated
`nova2.xml` remains unchanged (SHA256
`ef9fd2577701e0182280935301491a7a9ca97cfb44ac3d8f5ffb99c19ac83c1e`).
