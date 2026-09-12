#!/usr/bin/env python3
"""查清两边的 dtype：ninfer 的 format 字段 vs HF safetensors header 的 dtype。"""
import json, pathlib, struct
from collections import Counter

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}

idx = json.loads((HF / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdr = {}
def hf_meta(name):
    sh = wm.get(name)
    if not sh: return None
    if sh not in hdr:
        with open(HF / sh, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = json.loads(f.read(hn).decode("utf-8"))
    return hdr[sh].get(name)

pairs = [
    ("text/layers/0/input_norm", "model.language_model.layers.0.input_layernorm.weight"),
    ("text/layers/0/post_attention_norm", "model.language_model.layers.0.post_attention_layernorm.weight"),
    ("text/layers/0/gdn/norm", "model.language_model.layers.0.linear_attn.norm.weight"),
    ("text/layers/0/gdn/a_log", "model.language_model.layers.0.linear_attn.A_log"),
    ("text/layers/0/gdn/dt_bias", "model.language_model.layers.0.linear_attn.dt_bias"),
    ("text/layers/0/gdn/convolution", "model.language_model.layers.0.linear_attn.conv1d.weight"),
    ("text/final_norm", "model.language_model.norm.weight"),
    ("text/layers/3/attention/query_norm", "model.language_model.layers.3.self_attn.q_norm.weight"),
]
print("%-44s %-26s %-24s %s" % ("ninfer 名", "ninfer format/shape", "HF dtype/shape", "HF 名尾"))
for a, b in pairs:
    oa = objs.get(a)
    mb = hf_meta(b)
    sa = "%s %s" % (oa["format"], oa.get("shape")) if oa else "（无）"
    sb = "%s %s" % (mb["dtype"], mb["shape"]) if mb else "（无）"
    print("%-44s %-26s %-24s %s" % (a.split("/",1)[1][:42], sa, sb, b.split("layers.")[-1]))

print("\n=== HF 全部 dtype 分布 ===")
c = Counter()
for name in wm:
    m = hf_meta(name)
    if m: c[m["dtype"]] += 1
print("  %s" % dict(c))

print("\n=== ninfer 全部 format 分布 ===")
c2 = Counter(o["format"] for o in objs.values())
print("  %s" % dict(c2))

print("\n=== HF 里 embed_tokens（ninfer 是 text/token_embedding）===")
for k in wm:
    if "embed" in k:
        m = hf_meta(k)
        print("  %-58s %s %s" % (k, m["dtype"], m["shape"]))
