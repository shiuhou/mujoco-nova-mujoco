# Official X-Trainer USD Source

These files were downloaded from the public Dobot dataset:

- Dataset: `https://huggingface.co/datasets/Dobot-robot/X-trainer`
- Revision: `02f008252001f654e28dee40d615c713ff79935a`
- Dataset README: `x-trainer source files`

The relevant source paths are:

```text
x-trainer/assets/robots/x_trainer.usd
x-trainer/assets/robots/X-Trainer-URDF/urdf/
  x-trainer_asm-0226.SLDASM/
    x-trainer_asm-0226.SLDASM.usd
    configuration/x-trainer_asm-0226.SLDASM_base.usd
    configuration/x-trainer_asm-0226.SLDASM_physics.usd
    configuration/x-trainer_asm-0226.SLDASM_sensor.usd
```

The files are USD crate layers. The base layer contains the visual, collider,
and mesh scopes. It reports metres per unit and Z up. The original SolidWorks
assembly and editable CAD feature tree are not included in this download.

Downloaded SHA-256 hashes:

```text
c8c96c029f844682e2df76e219f5be004ce112c654da01fe78c2ce92d9add3d9  x_trainer.usd
3fe70ed88e720225471984e105fbe6e0955309c7e125391577e58fce26a820be  x-trainer_asm-0226.SLDASM.usd
c31afee5695353406e0b1cacf8b203bf2aeeb9a422d92e81834dcb5228e74f5b  x-trainer_asm-0226.SLDASM_base.usd
6b961adb90c4146b7008241c053e1ed92ee74cb63b850a2159ad521064372966  x-trainer_asm-0226.SLDASM_physics.usd
edd5cdbb73d24d6d5480e7be66f7b12112820574334f60438b98131339c215fd  x-trainer_asm-0226.SLDASM_sensor.usd
```

Use the upstream dataset terms when redistributing these assets.
