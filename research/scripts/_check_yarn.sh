#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) bf16_weights 的定义与取值（决定 offset=0 还是 1） ==="
grep -rn 'bf16_weights' "$R/src/targets/qwen3_6/impl/" "$R/src/targets/qwen3_6_27b/impl/" 2>/dev/null | head -10 | cut -c1-150
echo
echo "=== 2) dspark 的 rope 调用点（A5 说用纯 rope_theta） ==="
grep -n -B3 -A3 'ops::rope' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | head -24 | cut -c1-150
echo
echo "=== 3) checkpoint 的 rope_scaling 声明 ==="
python3 - <<'PY'
import json, pathlib
for p in ("/mnt/c/Users/User/Documents/ziqinzhang/tmp/dspark-config.json",
          "/mnt/c/Users/User/Documents/ziqinzhang/data/dspark-config.json"):
    f = pathlib.Path(p)
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    print("  file:", p)
    for k in ("rope_scaling", "rope_theta", "rope_parameters", "max_position_embeddings",
              "yarn", "factor", "original_max_position_embeddings"):
        if k in d:
            print("   %-32s %s" % (k, json.dumps(d[k], ensure_ascii=False)[:160]))
PY
echo
echo "=== 4) target(qwen3.8-27b) 侧的 yarn 配置与引擎 yarn 支持 ==="
grep -rn 'rope_yarn4\|yarn' "$R/include/ninfer/ops/rope.h" 2>/dev/null | head -6 | cut -c1-150
grep -rn 'yarn' "$R/src/targets/qwen3_6_27b/impl/config.h" 2>/dev/null | head -6 | cut -c1-140
