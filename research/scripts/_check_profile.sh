#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== variant.cpp 425-495（投影格式按 profile 选择处） ==="
sed -n '425,495p' "$R/src/targets/qwen3_6_27b/impl/variant.cpp"
echo
echo "=== artifact 里投影类张量的格式分布 ==="
python3 - <<'PY'
import collections
import json
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as h:
    h.read(8)
    (L,) = struct.unpack("<Q", h.read(8))
    doc = json.loads(h.read(L).decode())
cnt = collections.Counter()
for e in doc["objects"]:
    if e.get("kind") != "tensor" or not e["name"].startswith("text/"):
        continue
    n = e["name"]
    if any(k in n for k in ("in_proj", "qkv", "q_proj", "gate", "down", "up", "conv")):
        leaf = "/".join(n.split("/")[-2:])
        cnt[(leaf, e.get("format"))] += 1
for (leaf, fmt), v in sorted(cnt.items(), key=lambda x: -x[1])[:16]:
    print(f"   {leaf:<36} {fmt:<22} x{v}")
PY
