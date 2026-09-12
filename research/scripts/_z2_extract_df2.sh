#!/bin/bash
# Extract dflash2-related sources from the wheel to inspect requirements (no install needed).
W=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/df2_src
rm -rf "$OUT"; mkdir -p "$OUT"
python3 - "$W" "$OUT" <<'PYEOF'
import sys, zipfile, os
w, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(w)
targets = [
 "vllm/model_executor/models/qwen3_dflash2.py",
 "vllm/model_executor/models/qwen3_dflash.py",
 "vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py",
 "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
 "vllm/v1/spec_decode/dflash.py",
]
for t in targets:
    try:
        data = z.read(t)
    except KeyError:
        print("MISSING: " + t); continue
    dest = os.path.join(out, t.replace("/", "__"))
    open(dest, "wb").write(data)
    print("EXTRACTED %-70s %8d bytes" % (t, len(data)))
PYEOF
echo "=== extracted to $OUT ==="
ls -la "$OUT"
