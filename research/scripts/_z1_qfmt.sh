#!/bin/bash
for repo in "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090" "unsloth/Qwen3.8-27B-NVFP4" "RadixArk/Qwen3.8-27B-NVFP4" "cyankiwi/Qwen3.8-27B-AWQ-INT4"; do
  echo "################ $repo"
  safe=$(echo "$repo" | tr '/' '_')
  timeout 90 curl -sL "https://hf-mirror.com/$repo/resolve/main/config.json" -o "/tmp/cfg_$safe.json"
  python3 - "/tmp/cfg_$safe.json" <<'PYEOF'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception as e:
    print("  cfg err:", e); sys.exit()
print("  arch      =", c.get("architectures"), c.get("model_type"))
print("  quant     =", json.dumps(c.get("quantization_config"))[:600])
print("  hf_quant_cfg =", json.dumps(c.get("hf_quant_config"))[:300])
t = c.get("text_config", c)
print("  layers=%s hidden=%s lmo=%s" % (t.get("num_hidden_layers"), t.get("hidden_size"), c.get("language_model_only")))
PYEOF
done
