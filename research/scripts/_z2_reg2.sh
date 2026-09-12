#!/bin/bash
# How is the DFlash2 speculator selected? Look at spec_decode registries + speculative config.
W=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
python3 - "$W" <<'PYEOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()

def dump(path, grep=None, head=None, ctx=0):
    try:
        txt = z.read(path).decode('utf-8', 'replace')
    except KeyError:
        print("MISSING " + path); return
    lines = txt.splitlines()
    print("")
    print("########## %s (%d lines) ##########" % (path, len(lines)))
    if grep:
        for i, ln in enumerate(lines):
            if any(g in ln for g in grep):
                lo = max(0, i-ctx); hi = min(len(lines), i+ctx+1)
                for j in range(lo, hi):
                    print("  %5d: %s" % (j+1, lines[j][:170]))
                if ctx: print("  ---")
    elif head:
        for i, ln in enumerate(lines[:head], 1):
            print("  %5d: %s" % (i, ln[:170]))

# 1) registry context around DFlash2
dump("vllm/model_executor/models/registry.py", grep=["DFlash", "DSpark", "DraftModel"], ctx=0)

# 2) speculator dispatch
dump("vllm/v1/worker/gpu/spec_decode/__init__.py", head=200)
dump("vllm/v1/worker/gpu/spec_decode/dflash/__init__.py", head=60)

# 3) speculative config: valid method strings
dump("vllm/config/speculative.py", grep=["dflash", "DFlash", "dflash2", "method", "SPECUL"], ctx=1)
PYEOF
