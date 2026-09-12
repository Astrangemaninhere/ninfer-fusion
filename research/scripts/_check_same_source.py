#!/usr/bin/env python3
"""零成本同源判定：读 .ninfer 头部 JSON，与下载/本地的 HF config.json 逐项比。
.ninfer 容器: magic[0:8], 偏移 8 处 u64(LE) JSON 长度, JSON 从偏移 16 开始。
"""
import json, pathlib, struct, sys

def ninfer_header(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    try:
        j = json.loads(raw.decode("utf-8", "replace"))
    except Exception as e:
        return {"_magic": magic.hex(), "_json_bytes": n, "_parse_error": str(e)}
    return {"_magic": magic.hex(), "_json_bytes": n, **j}

def summarize_hf(p):
    j = json.loads(pathlib.Path(p).read_text(encoding="utf-8", errors="replace"))
    keys = ["architectures","model_type","num_hidden_layers","hidden_size",
            "intermediate_size","num_attention_heads","num_key_value_heads",
            "head_dim","vocab_size","rope_theta","tie_word_embeddings","max_position_embeddings"]
    return {k: j.get(k) for k in keys if k in j}

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
cands = [
    ("ninfer_dspark",  "/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer"),
    ("ninfer_base",    "/home/user/models/qwen3_8_27b_nvfp4.ninfer"),
    ("ninfer_dflash2", "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"),
]
for tag, p in cands:
    if not pathlib.Path(p).exists():
        print("[%s] 缺 %s" % (tag, p)); continue
    h = ninfer_header(p)
    print("=== %s ===" % tag)
    print("  magic=%s  header_json=%d B" % (h.get("_magic"), h.get("_json_bytes", 0)))
    for k in ("identity", "config", "text_config", "geometries", "format", "quantization", "version"):
        if k in h:
            v = h[k]
            s = json.dumps(v, ensure_ascii=False)
            print("  %-14s %s" % (k, s[:300]))
    other = [k for k in h if not k.startswith("_") and k not in
             ("identity","config","text_config","geometries","format","quantization","version")]
    print("  其它键: %s" % ", ".join(sorted(other)[:12]))
    print()

print("=== 下载的目标 config.json ===")
tp = J / "models/Qwen3.8-27B-NVFP4-RTX5090/config.json"
if tp.exists():
    s = summarize_hf(tp)
    print(json.dumps(s, ensure_ascii=False, indent=2))
else:
    print("  缺 %s" % tp)
print()
print("=== 破限版目录（若存在 HF 权重）===")
for d in sorted((J / "models").glob("*Abliterated*")):
    hf = list(d.glob("*.safetensors"))
    print("  %s  safetensors=%d  %s" % (d.name, len(hf), ", ".join(x.name for x in hf[:3])))
