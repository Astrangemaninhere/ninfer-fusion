#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== A5 审计用的 dspark 源：models/qwen3.8-27b-dspark-zh/ ==="
ls -l "$D/models/qwen3.8-27b-dspark-zh/" 2>&1 | head -8
echo "--- 它的 config.json 关键字段 ---"
python3 - <<'PY' 2>&1
import json, pathlib
for p in [r"/mnt/c/Users/User/Documents/ziqinzhang/models/qwen3.8-27b-dspark-zh/config.json",
          r"/mnt/c/Users/User/Documents/ziqinzhang/data/draft_model/config.json"]:
    f = pathlib.Path(p)
    print(f"--- {p.split('/')[-2]}/config.json ---")
    if not f.exists():
        print("   MISSING"); continue
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        print("   parse error:", exc); continue
    for k in ("architectures", "num_attention_heads", "num_key_value_heads", "head_dim",
              "num_hidden_layers", "intermediate_size", "rope_parameters", "mask_token_id",
              "target_layer_ids", "block_size", "markov_rank", "dflash_config",
              "sliding_window", "num_target_layers", "sample_from_anchor"):
        if k in d:
            print(f"   {k} = {d[k]}")
PY
echo
echo "=== 我们引擎里 dspark(DFlashConfig) 的对应值 ==="
sed -n '100,120p' "$D/../ziqinzhang/ninfer-upstream/src/targets/qwen3_6_27b/impl/config.h" 2>/dev/null | head -5
grep -n "mask_token\|target_feature_layers\|rope_theta\|query_heads\|local_window\|markov_rank" /mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream/src/targets/qwen3_6_27b/impl/config.h 2>/dev/null | head -20
