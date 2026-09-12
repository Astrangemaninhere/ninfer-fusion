#!/usr/bin/env python3
"""修掉上一步的连带错误：optional 自己的 emplace 被误改成 `->emplace`。"""
import pathlib

P = pathlib.Path("/home/user/ninfer-fusion/src/runtime/engine/engine_core.h")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
n = 0
for i, l in enumerate(lines):
    if "->emplace(" in l and ("head_inspection" in l or "candidate_inspection" in l):
        lines[i] = l.replace("->emplace(", ".emplace(")
        n += 1
P.write_text("".join(lines), encoding="utf-8")
print("修正 %d 行 ->emplace" % n)

# 打印两处插入区域，人工核对
txt = "".join(lines).splitlines()
for lo, hi, tag in ((1716, 1736, "head"), (1795, 1815, "candidate")):
    print("\n--- %s (%d-%d) ---" % (tag, lo, hi))
    for i in range(lo - 1, min(hi, len(txt))):
        print("  %4d %s" % (i + 1, txt[i][:118]))
