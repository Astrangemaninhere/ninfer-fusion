#!/bin/bash
# Inspect the HF-format dflash2 config that sits next to tmp/dflash2-fp8/model.safetensors
C=/mnt/c/Users/User/Documents/ziqinzhang/tmp/dflash2-fp8-config.json
echo "=== file ==="
ls -la "$C"
echo ""
echo "=== FULL CONTENT ==="
cat "$C"
echo ""
echo "=== STRUCTURED SUMMARY ==="
python3 - "$C" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("architectures", "model_type", "num_hidden_layers", "hidden_size",
          "num_attention_heads", "num_key_value_heads", "vocab_size",
          "max_position_embeddings", "block_size", "mask_token_id",
          "num_target_layers", "target_hidden_size", "target_layer_ids",
          "use_sliding_window", "sliding_window"):
    if k in d:
        print("  %-26s %s" % (k, d[k]))
print("  has dflash_config:", "dflash_config" in d)
if "dflash_config" in d:
    print("  dflash_config =", json.dumps(d["dflash_config"], indent=2))
qc = d.get("quantization_config")
if qc:
    print("  quantization_config.quant_method:", qc.get("quant_method"))
    print("  quantization_config.fmt:", qc.get("fmt"))
    print("  modules_to_not_convert count:", len(qc.get("modules_to_not_convert", [])))
PYEOF
echo ""
echo "=== dir listing of tmp/dflash2-fp8 (is config.json inside?) ==="
ls -la /mnt/c/Users/User/Documents/ziqinzhang/tmp/dflash2-fp8/
echo ""
echo "=== sibling dflash2-fp8-* files ==="
ls -la /mnt/c/Users/User/Documents/ziqinzhang/tmp/ | grep -i 'dflash2-fp8'
