#!/bin/bash
# Inspect vllm 0.29.0 wheel contents for dflash2 using python zipfile (no unzip binary in WSL).
W=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
python3 - "$W" <<'PYEOF'
import sys, zipfile
w = sys.argv[1]
z = zipfile.ZipFile(w)
names = z.namelist()
print("TOTAL_ENTRIES=%d" % len(names))

def show(label, pat):
    hits = [n for n in names if pat in n.lower()]
    print("")
    print("=== %s : %d hits ===" % (label, len(hits)))
    for h in hits[:60]:
        print("  " + h)
    return hits

show("dflash2 (path)", "dflash2")
show("dflash (path, any)", "dflash")
show("dspark (path)", "dspark")

spec = [n for n in names if "spec_decode" in n.lower()]
print("")
print("=== spec_decode paths : %d ===" % len(spec))
for s in spec[:80]:
    print("  " + s)

q3 = [n for n in names if n.startswith("vllm/model_executor/models/") and "qwen3" in n.lower()]
print("")
print("=== models/qwen3* : %d ===" % len(q3))
for s in sorted(q3):
    print("  " + s)

# also check METADATA for version
meta = [n for n in names if n.endswith(".dist-info/METADATA")]
print("")
print("=== dist-info ===")
for m in meta:
    print("  " + m)
    txt = z.read(m).decode("utf-8", "replace")
    for line in txt.splitlines()[:12]:
        print("     " + line)
    for line in txt.splitlines():
        if line.startswith("Requires-Dist: torch") or line.startswith("Requires-Dist: flashinfer"):
            print("     DEP> " + line)
PYEOF
