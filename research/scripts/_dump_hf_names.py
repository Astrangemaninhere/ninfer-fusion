#!/usr/bin/env python3
"""只打印 HF 侧名字结构，供映射用。"""
import json, pathlib
HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")
idx = json.loads((HF / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = list(idx["weight_map"])
print("HF 张量总数 %d" % len(wm))
print("\n=== 第 0 层（应是 linear_attn）===")
for k in sorted(x for x in wm if x.startswith("model.language_model.layers.0.")):
    print("   %s" % k)
print("\n=== 第 3 层（应为 full attention，作对照）===")
for k in sorted(x for x in wm if x.startswith("model.language_model.layers.3.")):
    print("   %s" % k)
print("\n=== 顶层（非 layers）===")
for k in sorted(x for x in wm if ".layers." not in x):
    print("   %s" % k)
print("\n=== 层类型分布（从 config）===")
c = json.loads((HF / "config.json").read_text(encoding="utf-8"))
tc = c.get("text_config", c)
print("  num_hidden_layers=%s hidden=%s vocab=%s" % (tc.get("num_hidden_layers"), tc.get("hidden_size"), tc.get("vocab_size")))
lt = tc.get("layer_types")
if lt:
    from collections import Counter
    print("  layer_types 计数: %s" % dict(Counter(lt)))
    print("  前 20: %s" % lt[:20])
print("\n=== 量化配置 ===")
q = json.loads((HF / "hf_quant_config.json").read_text(encoding="utf-8"))
print("  %s" % json.dumps(q, ensure_ascii=False)[:400])
