#!/usr/bin/env python3
"""Instantiation inventory of the g35/muse/main GQA decode objects.

Reads `nm` output (mangled names, kept intact), normalizes `__device_stub__`
prefix and `.cold` clone suffixes, demangles with c++filt, and counts unique
kernel instantiations per kernel template family. Read-only: runs nm + c++filt.
"""
import re
import subprocess
import sys
from collections import Counter, defaultdict

B = "/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher"
TARGETS = {
    "g35":  f"{B}/gqa_attention_decode_g35.cu.o",
    "muse": f"{B}/gqa_attention_decode_muse.cu.o",
    "main": f"{B}/gqa_attention_decode.cu.o",
}

def run(cmd, inp=None):
    return subprocess.run(cmd, input=inp, capture_output=True, text=True,
                          errors="replace").stdout

def demangle(names):
    if not names:
        return []
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True,
                       text=True, errors="replace").stdout.splitlines()
    return p

def norm(sym, typ):
    s = sym
    if s.startswith("__device_stub__"):
        s = s[len("__device_stub__"):]
    s = re.sub(r" \[clone \.cold\]$", "", s)
    s = re.sub(r" \[clone [^\]]*\]$", "", s)
    if s.startswith("__sti__") or s.startswith("____nv") or s.startswith("__nv_") \
       or s.startswith("__cuda") or s.startswith("_GLOBAL__") or s.startswith("_Z"):
        # keep _Z* (mangled) but drop registration glue handled below
        pass
    return s

def inventory(path, tag):
    out = run(["nm", "--defined-only", path])
    rows = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        addr, typ, name = parts
        if typ not in ("T", "t", "W", "w"):
            continue
        rows.append((typ, name))
    # group unique by normalized name; strip clone suffixes to find duplicates
    norm_names = [norm(n, t) for t, n in rows]
    uniq = sorted(set(norm_names))
    dm = demangle([n for n in uniq if n.startswith("_Z") or "ZN" in n or "_ZN" in n])
    dm_map = dict(zip([n for n in uniq if n.startswith("_Z") or "ZN" in n or "_ZN" in n], dm))
    # kernel families: name up to '<'
    fam = Counter()
    fam_args = defaultdict(set)
    for n in uniq:
        d = dm_map.get(n, n)
        base = d.split("<")[0].split("(")[0].strip()
        m = re.search(r"([A-Za-z_0-9:]+kernel(?:[A-Za-z_0-9]*))<", d)
        key = m.group(1) if m else base
        args = d[d.find("<")+1:d.rfind(">")] if "<" in d else ""
        fam[key] += 1
        fam_args[key].add(args)
    print(f"\n########## {tag}: {path.split('/')[-1]} ##########")
    print(f"defined T/t/W/w symbols: {len(rows)}   unique (norm): {len(uniq)}")
    print("--- kernel families: #unique instantiation signatures ---")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        if "kernel" in k:
            print(f"  {v:4d}  {k}")
    print("--- distinct template-arg tuples per family (truncated) ---")
    for k in sorted(fam_args):
        if "kernel" not in k:
            continue
        args = sorted(fam_args[k])
        if len(args) <= 24:
            for a in args:
                print(f"    [{k}] <{a}>")
        else:
            print(f"    [{k}] {len(args)} distinct arg tuples; first 8:")
            for a in args[:8]:
                print(f"       <{a}>")
    other = [k for k in fam if "kernel" not in k]
    print(f"--- non-kernel defined symbols: {len(other)} ---")
    for k in sorted(other)[:12]:
        print(f"    {k}  (x{fam[k]})")
    return fam

for tag, path in TARGETS.items():
    try:
        inventory(path, tag)
    except FileNotFoundError:
        print(f"{tag}: nm not found")
    except Exception as e:
        print(f"{tag}: ERROR {e}")
