#!/usr/bin/env python3
"""Compact instantiation-grid digest per kernel family (read-only: nm + c++filt)."""
import re
import subprocess
from collections import defaultdict

B = "/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher"
TARGETS = [
    ("g35",  f"{B}/gqa_attention_decode_g35.cu.o"),
    ("muse", f"{B}/gqa_attention_decode_muse.cu.o"),
    ("main", f"{B}/gqa_attention_decode.cu.o"),
]

def nm_names(path):
    out = subprocess.run(["nm", "--defined-only", path], capture_output=True,
                         text=True, errors="replace").stdout
    names = []
    for line in out.splitlines():
        p = line.split(" ", 2)
        if len(p) == 3 and p[1] in ("T", "t", "W", "w"):
            names.append(p[2])
    return names

def normalize(names):
    res = []
    for n in names:
        n = re.sub(r" \[clone [^\]]*\]$", "", n)
        if n.startswith("__device_stub__"):
            n = n[15:]
        if any(n.startswith(p) for p in ("__sti__", "__nv_", "____nv", "__cuda", "_GLOBAL__")):
            continue
        if "cudaRegister" in n:
            continue
        res.append(n)
    return res

def demangle(names):
    if not names:
        return []
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True,
                       text=True, errors="replace").stdout
    lines = p.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if len(lines) != len(names):
        lines = [subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
                 for n in names]
    return lines

def split_args(s):
    """split top-level comma args"""
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args

for tag, path in TARGETS:
    names = normalize(nm_names(path))
    d = demangle(names)
    sigs = sorted({x.split("(")[0].strip() for x in d})
    fam = defaultdict(set)
    for s in sigs:
        m = re.match(r"^(.*?::[A-Za-z_0-9]*kernel)\s*<(.*)>\s*$", s)
        if not m:
            fam["[non-kernel] " + s].add("")
            continue
        fam[m.group(1)].add(m.group(2))
    print(f"\n===== {tag} : {path.split('/')[-1]} =====")
    for base in sorted(fam, key=lambda b: -len(fam[b])):
        argsets = [split_args(a) for a in sorted(fam[base]) if a]
        short = base.rsplit("::", 1)[-1]
        if not argsets:
            print(f"  {len(fam[base]):3d}  {short}   (no args)")
            continue
        npos = max(len(a) for a in argsets)
        cols = []
        for i in range(npos):
            vals = sorted({a[i] for a in argsets if len(a) > i},
                          key=lambda v: (len(v), v))
            # compact long geometry lists
            def shortv(v):
                v = v.replace("ninfer::ops::", "").replace("GqaGeometry<", "G<").replace(">", ">")
                return v
            shown = vals if len(vals) <= 8 else vals[:8] + [f"...(+{len(vals)-8})"]
            cols.append(f"p{i+1}=" + ",".join(shortv(v) for v in shown))
        print(f"  {len(fam[base]):3d}  {short}")
        for c in cols:
            print(f"        {c}")
