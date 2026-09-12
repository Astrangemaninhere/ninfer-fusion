#!/usr/bin/env python3
"""扫全树：找出"#include 落在 namespace 内部"的写法（这一类会让系统头被吞进命名空间，
报出 'memchr has not been declared in ::' 之类的怪错）。S28 的 decoder_state.cpp 就是一例。"""
import pathlib
import re
import subprocess
import sys

R = pathlib.Path("/home/user/ninfer-fusion/src")
hits = []
for p in list(R.rglob("*.cpp")) + list(R.rglob("*.h")) + list(R.rglob("*.cuh")) + list(R.rglob("*.cu")):
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        continue
    depth = 0
    first_ns = None
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        if re.match(r"^namespace\b", s):
            if depth == 0 and first_ns is None:
                first_ns = i
            # 单行 `namespace { ... }` 自闭合：两个都要算，否则会假阳性
            depth += s.count("{") - s.count("}")
        if s.startswith("#include") and first_ns is not None and depth > 0:
            hits.append((p, i, s))
            break

print("命中 %d 个文件（include 出现在已打开的 namespace 内）：" % len(hits))
for p, i, s in hits:
    print("  %s:%d  %s" % (p.relative_to(R.parent), i, s[:90]))

# 自动修：把该 include 移到文件顶部（最后一个顶层 #include 之后）
if "--fix" in sys.argv:
    fixed = 0
    for p, i, s in hits:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        inc = lines[i - 1]
        del lines[i - 1]
        last = 0
        for k, l in enumerate(lines[:i - 1]):
            if l.startswith("#include"):
                last = k
        lines.insert(last + 1, inc)
        p.write_text("".join(lines), encoding="utf-8")
        fixed += 1
        print("  已移动: %s" % p.name)
    print("修复 %d 个文件" % fixed)
