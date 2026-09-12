#!/usr/bin/env python3
"""把两处守卫的缩进对齐（纯排版，配合周围代码风格）。"""
import pathlib

P = pathlib.Path("/home/user/ninfer-fusion/src/runtime/engine/engine_core.h")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
fixed = 0
for i, l in enumerate(lines):
    if l.strip().startswith("if (!head_inspection) {") or l.strip().startswith("if (!candidate_inspection) {"):
        base = "            " if "head_inspection" in l else "                "
        lines[i] = base + l.strip() + "\n"
        for k in (1, 2, 3):
            if i + k < len(lines):
                lines[i + k] = base + "    " + lines[i + k].strip() + "\n"
        fixed += 1
P.write_text("".join(lines), encoding="utf-8")
print("对齐 %d 处守卫缩进" % fixed)
