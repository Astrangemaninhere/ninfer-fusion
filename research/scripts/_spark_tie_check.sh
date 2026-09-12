#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
S=$J/models/Spark-X2.5-4B
echo "=== verify the tie premise from the real index ==="
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Spark-X2.5-4B/model.safetensors.index.json")
d = json.loads(p.read_text())
wm = d["weight_map"]
print("  total tensors:", len(wm), " total bytes:", d.get("metadata", {}).get("total_size"))
names = sorted(wm)
print("  has lm_head.weight      :", "lm_head.weight" in wm)
print("  has embed_tokens.weight :", any("embed_tokens" in n for n in names))
print("  layer_0 keys:", [n.split(".")[-1] for n in names if n.startswith("model.layers.0.")])
print("  non-layer keys:", [n for n in names if "layers." not in n][:12])
PY
echo
echo "=== what does the converter do with lm_head / tie today? ==="
R=$J/ninfer-fusion-repo
grep -rn 'lm_head\|tie_word' "$R/tools/convert/qwen3_6_27b/convert.py" 2>/dev/null | head -14 | cut -c1-140
echo
echo "=== Spark weight download progress ==="
tail -4 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-130
du -sh "$S" 2>/dev/null
