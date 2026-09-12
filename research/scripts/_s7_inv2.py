#!/usr/bin/env python3
"""Precise instantiation inventory for GQA decode objects (read-only: nm + c++filt)."""
import re
import subprocess
from collections import Counter, defaultdict

B = "/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher"
TARGETS = [
    ("g35",  f"{B}/gqa_attention_decode_g35.cu.o"),
    ("muse", f"{B}/gqa_attention_decode_muse.cu.o"),
    ("main", f"{B}/gqa_attention_decode.cu.o"),
]

def nm_defined_text(path):
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
            n = n[len("__device_stub__"):]
        if n.startswith("__sti__") or n.startswith("__nv_") or n.startswith("____nv") \
           or n.startswith("__cuda") or n.startswith("_GLOBAL__") or "cudaRegister" in n:
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
        # fall back: per-name demangle
        lines = [subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
                 for n in names]
    return lines

for tag, path in TARGETS:
    names = normalize(nm_defined_text(path))
    d = demangle(names)
    sig = sorted({x.split("(")[0].strip() for x in d})
    fam = defaultdict(list)
    for s in sig:
        m = re.match(r"^(.*?\b([A-Za-z_0-9]+kernel))\s*<", s)
        if m:
            base = m.group(1)
            args = s[m.end():]
            args = args[:args.rfind(">")]
            fam[base].append(args)
        else:
            fam[s].append("")
    print(f"\n### {tag}: {path.split('/')[-1]}")
    print(f"    raw T/t/W/w={len(names)}  unique-signatures={len(sig)}")
    for base in sorted(fam, key=lambda b: -len(fam[b])):
        args = fam[base]
        if "kernel" not in base:
            print(f"    [non-kernel] {base}")
            continue
        print(f"    {len(args):4d}x  {base.rsplit('::',1)[-1]}")
        for a in sorted(set(args)):
            if a:
                print(f"           <{a}>")
