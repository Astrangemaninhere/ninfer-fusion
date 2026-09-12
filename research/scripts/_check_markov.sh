#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== 产物里 dspark 的 markov 张量（dflash/ 命名空间）==='
python3 - <<'PY'
import json, struct
P="/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer"
with open(P,"rb") as f:
    f.read(8); n=struct.unpack("<Q",f.read(8))[0]
    man=json.loads(f.read(n).decode("utf-8","replace"))
ts=[o for o in man["objects"] if o.get("kind")=="tensor"]
print("  总张量 %d" % len(ts))
for o in ts:
    nm=o["name"]
    if nm.startswith("dflash/") and ("markov" in nm.lower() or "w1" in nm or "w2" in nm or "rank" in nm):
        print("   %-52s %-20s %s" % (nm, o["format"], o.get("shape")))
print("  --- dflash/ 顶层件（非 layers/）---")
for o in ts:
    nm=o["name"]
    if nm.startswith("dflash/") and "/layers/" not in nm:
        print("   %-52s %-20s %s" % (nm, o["format"], o.get("shape")))
PY
echo
echo '=== 引擎侧 markov 的绑定与用法 ==='
grep -rn 'markov' $R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp 2>/dev/null | head -12 | cut -c1-150
echo
echo '=== markov kernel 的调用点与缩放 ==='
grep -rn 'dspark_markov_argmax\|markov_scale\|kDsparkMarkovRank' $R/src --include=*.h --include=*.cu --include=*.cpp 2>/dev/null | grep -v '\.orig' | head -12 | cut -c1-150
