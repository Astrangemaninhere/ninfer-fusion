import json, struct, sys

PATH = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(PATH, "rb") as f:
    magic = f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    raw = f.read(jlen)
print("magic =", magic)
print("json 长度 =", jlen)
doc = json.loads(raw.decode("utf-8"))
objs = doc.get("objects") or doc.get("tensors") or []
print("对象总数 =", len(objs))
print()
print("=== dflash2/* 与 selector/codebook/draft_head 相关对象 ===")
keys = ("dflash2/", "selector", "codebook", "draft_head", "output_head")
for o in objs:
    name = str(o.get("name", ""))
    if any(k in name for k in keys):
        shp = o.get("shape")
        print(f"  {name:<52} kind={o.get('kind')} fmt={o.get('format')} shape={shp}")
print()
print("=== 顶层 JSON 键 ===")
print(sorted(doc.keys()))
for k in ("vocab_size", "token_domain", "hidden", "selector_rank", "selector_top_k",
          "draft_head_rows", "num_hidden_layers", "arch"):
    if k in doc:
        print(f"  {k} = {doc[k]}")
