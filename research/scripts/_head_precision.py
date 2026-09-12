import json, pathlib
# 1) 源模型（HF）里 lm_head 的精度角色
cfg = json.loads(pathlib.Path("/home/user/models/q3nvfp4/config.json").read_text())
qc = cfg.get("quantization_config") or {}
ign = qc.get("ignore") or []
print("=== 源 HF config: quantization_config ===")
print("  quant_method =", qc.get("quant_method"))
print("  ignore 条目数 =", len(ign))
hits = [x for x in ign if any(k in str(x) for k in ("lm_head", "head", "embed", "mtp"))]
print("  与头/嵌入/MTP 相关的 ignore 条目:")
for h in hits:
    print("    ", h)
print()
print("=== config 里是否另有 lm_head 精度声明 ===")
for k in ("torch_dtype", "dtype", "text_config"):
    if k in cfg:
        v = cfg[k]
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={v[kk]}" for kk in ("torch_dtype","dtype","num_hidden_layers","hidden_size","vocab_size","max_position_embeddings") if kk in v))
        else:
            print(f"  {k} = {v}")
print()
# 2) 我们 artifact 里输出头/草稿头的格式
import struct
p = pathlib.Path("/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer")
with p.open("rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8)); doc = json.loads(f.read(jlen).decode())
objs = {o["name"]: o for o in doc["objects"]}
print("=== 我们 artifact 里的头相关对象格式 ===")
for n, o in objs.items():
    if any(k in n for k in ("output_head", "draft_head", "embed", "mtp", "codebook", "hidden_projection")):
        print(f"  {n:<52} fmt={o.get('format'):<28} shape={o.get('shape')}")
print()
print("=== artifact 全部 format 直方图 ===")
from collections import Counter
c = Counter(o.get("format") for o in doc["objects"])
for k, v in c.most_common():
    print(f"  {str(k):<30} {v}")
# 3) q3nvfp4 的 hf_quant_config 对照
hqp = pathlib.Path("/home/user/models/q3nvfp4/hf_quant_config.json")
if hqp.exists():
    hq = json.loads(hqp.read_text())
    print()
    print("=== hf_quant_config.json ===")
    for k in ("quantization", "quant_algo", "exclude_modules", "kv_cache_quant_algo"):
        if k in hq:
            v = hq[k]
            if isinstance(v, dict):
                print(f"  {k}: " + json.dumps(v, ensure_ascii=False)[:300])
            else:
                print(f"  {k} = {str(v)[:300]}")
