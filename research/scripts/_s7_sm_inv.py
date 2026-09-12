#!/usr/bin/env python3
import re, subprocess
from collections import defaultdict

OBJ = "/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/small_t.cu.o"
OBJ_FP8 = "/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/small_t_fp8.cu.o"

def nm_names(path):
    out = subprocess.run(["nm", "--defined-only", path], capture_output=True, text=True,
                         errors="replace").stdout
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
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                       errors="replace").stdout.split("\n")
    if p and p[-1] == "":
        p = p[:-1]
    if len(p) != len(names):
        p = [subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip() for n in names]
    return p

def split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "<": depth += 1
        elif ch == ">": depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): args.append(cur.strip())
    return args

for tag, path in (("small_t(causal)", OBJ), ("small_t_fp8(causal)", OBJ_FP8)):
    names = normalize(nm_names(path))
    d = demangle(names)
    sigs = sorted({x.split("(")[0].strip() for x in d})
    fam = defaultdict(set)
    nk = 0
    for s in sigs:
        m = re.match(r"^(.*?::[A-Za-z_0-9]*(?:kernel))\s*<(.*)>\s*$", s)
        if m:
            fam[m.group(1)].add(m.group(2))
        else:
            nk += 1
            fam["[non-kernel] " + s].add("")
    print(f"\n===== {tag} =====")
    print(f"raw defined text syms: {len(names)}  unique signatures: {len(sigs)}  non-kernel: {nk}")
    for base in sorted(fam, key=lambda b: -len(fam[b])):
        if base.startswith("[non-kernel]"):
            print(f"    {base}")
            continue
        short = base.rsplit("::", 1)[-1]
        argsets = [split_args(a) for a in sorted(fam[base]) if a]
        npos = max((len(a) for a in argsets), default=0)
        print(f"  {len(fam[base]):4d}x  {short}")
        for i in range(npos):
            vals = sorted({a[i] for a in argsets if len(a) > i}, key=lambda v: (len(v), v))
            shown = vals if len(vals) <= 10 else vals[:10] + [f"...(+{len(vals)-10})"]
            print(f"        p{i+1}=" + ",".join(v.replace("ninfer::ops::", "") for v in shown))
