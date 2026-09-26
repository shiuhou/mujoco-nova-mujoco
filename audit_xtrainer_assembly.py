#!/usr/bin/env python3
"""Audit the composed official X-Trainer USD and write a provenance report."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom

ROOT = "/x_trainer_asm_0226_SLDASM"
PARTS = ["base_link"] + [f"J{i}_{j}" for i in range(1, 3) for j in range(1, 9)]


def _value(prim: Usd.Prim, name: str):
    prop = prim.GetProperty(name)
    if not prop:
        return None
    if isinstance(prop, Usd.Relationship):
        return [str(p) for p in prop.GetTargets()]
    value = prop.Get()
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return str(value)
    return value


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value).replace("|", "\\|")


def build_report(stage_path: Path) -> str:
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open {stage_path}")
    cache = UsdGeom.XformCache()
    lines = [
        "# X-Trainer End-Effector Assembly Audit",
        "",
        f"Source: `{stage_path}`",
        "",
        "## Scope and frame boundary",
        "",
        "The composed `physics.usd` stage is audited in its X-Trainer assembly/world frame. "
        "Mesh prims are stage-level `/visuals/.../mesh` assets; component bodies are "
        "flat children under the assembly root. No prim is named `NovaFlange`, `TCP`, "
        "or `gripper`, so a Nova flange transform is not asserted from names alone.",
        "",
        "## Hierarchy (actual prim paths)",
        "",
        "```text",
        f"{ROOT}",
        "  /base_link",
        "  /J1_1 -> /J1_2 -> /J1_3 -> /J1_4 -> /J1_5 -> /J1_6",
        "      /J1_6 -> /J1_7 (prismatic finger) and /J1_8 (prismatic finger)",
        "  /J2_1 -> /J2_2 -> /J2_3 -> /J2_4 -> /J2_5 -> /J2_6",
        "      /J2_6 -> /J2_7 (prismatic finger) and /J2_8 (prismatic finger)",
        "  /joints/J1_1 ... /J4_6 (physics joints)",
        "```",
        "",
        "The arrows above are joint `body0 -> body1` relationships, not USD parent nesting: "
        "all component bodies are direct children of the assembly root.",
        "",
        "## Component transforms and meshes",
        "",
        "| Prim | Parent | local translate (m) | world translate (m) | visual mesh | APIs |",
        "|---|---|---:|---:|---|---|",
    ]
    for name in PARTS:
        prim = stage.GetPrimAtPath(f"{ROOT}/{name}")
        if not prim:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        local = _value(prim, "xformOp:translate")
        world = [float(x) for x in matrix.ExtractTranslation()]
        mesh = f"/visuals/{name}/{name}/mesh"
        has_mesh = bool(stage.GetPrimAtPath(mesh))
        lines.append(
            f"| `{prim.GetPath()}` | `{prim.GetParent().GetPath()}` | `{_fmt(local)}` | "
            f"`{_fmt(world)}` | `{mesh if has_mesh else '-'}` | "
            f"`{','.join(prim.GetAppliedSchemas()) or '-'}` |"
        )

    lines += [
        "",
        "## Joint and articulation data",
        "",
        "The physics layer contains no mimic relationship. J1/J2 joints 1-6 are revolute; "
        "J1/J2 joints 7-8 are prismatic. All reported values below are authored USD values.",
        "",
        "| Joint prim | type | body0 -> body1 | axis | lower | upper | localPos0 | localPos1 | drive |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ]
    for group in ("J1", "J2"):
        for index in range(1, 9):
            path = f"{ROOT}/joints/{group}_{index}"
            prim = stage.GetPrimAtPath(path)
            if not prim:
                continue
            typ = prim.GetTypeName()
            body0 = _value(prim, "physics:body0")
            body1 = _value(prim, "physics:body1")
            drive = "linear" if "Prismatic" in typ else "angular"
            lines.append(
                f"| `{path}` | `{typ}` | `{body0} -> {body1}` | `{_fmt(_value(prim, 'physics:axis'))}` | "
                f"{_fmt(_value(prim, 'physics:lowerLimit'))} | {_fmt(_value(prim, 'physics:upperLimit'))} | "
                f"`{_fmt(_value(prim, 'physics:localPos0'))}` | `{_fmt(_value(prim, 'physics:localPos1'))}` | `{drive}` |"
            )

    lines += [
        "",
        "## Assembly interpretation",
        "",
        "- Directly sourced: J1_6/J1_7/J1_8 and J2_6/J2_7/J2_8 geometry, body transforms, "
        "joint axes, limits, body relationships, and linear drive metadata.",
        "- The J1_6/J2_6 meshes are the larger distal body components and are extracted as "
        "body candidates; the USD provides no semantic proof that either is a Nova flange adapter.",
        "- J1_7/J1_8 and J2_7/J2_8 are opposing prismatic finger pairs. Their axis is USD `Z`; "
        "J1_7/J2_7 limits are [-0.04, 0] m and J1_8/J2_8 limits are [0, 0.04] m.",
        "- No mimic relationship, coupled actuator rule, or explicit nominal/open/closed state was found.",
        "- `T_flange_gripper = inverse(T_world_flange) @ T_world_gripper` is **not computed**: "
        "the stage has no identifiable Nova wrist/flange prim. The X-Trainer world frame and "
        "Nova2 Link6 frame are unrelated until measured or sourced from a Nova-mounted assembly.",
        "",
        "## Quantities requiring physical measurement",
        "",
        "Nova flange-to-body origin/orientation, tool/TCP frame, actual finger zero/open state, "
        "mechanical coupling between the two fingers, and whether this X-Trainer end-effector "
        "is the same hardware revision as the Nova gripper.",
        "",
        "## Provenance",
        "",
        "This report is generated from the pinned official USD physics composition. It does not "
        "modify `nova2.xml`, apply X-Trainer world transforms to Nova2, or claim physical fit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("gripper_assembly_report.md")
    )
    args = parser.parse_args()
    args.output.write_text(build_report(args.stage.resolve()), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
