#!/bin/bash
# Find how DFlash2 is registered / selected in the wheel.
W=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
python3 - "$W" <<'PYEOF'
import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()

# Files likely to hold registration / method dispatch
cands = [n for n in names if n.endswith('.py') and any(k in n for k in (
    'model_executor/models/registry.py',
    'config/speculative.py',
    'spec_decode/dflash.py',
    'models/qwen3_dflash.py',
    'models/qwen3.py',
    'config/__init__.py',
))]
print("=== scanning %d candidate files ===" % len(cands))
pats = [
    r'DFlash2', r'dflash2', r'DFlash', r'dflash2_speculator',
    r'register_model\(', r'method',
]
for n in sorted(cands):
    txt = z.read(n).decode('utf-8', 'replace')
    lines = txt.splitlines()
    hits = []
    for i, ln in enumerate(lines, 1):
        if ('DFlash2' in ln) or ('dflash2' in ln):
            hits.append((i, ln.rstrip()))
    if hits:
        print("")
        print("----- %s (%d hits) -----" % (n, len(hits)))
        for i, ln in hits[:40]:
            print("  %5d: %s" % (i, ln[:160]))
PYEOF
