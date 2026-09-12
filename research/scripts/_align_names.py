#!/usr/bin/env python3
"""对齐两边命名：打印 ninfer 与 HF 在"第 0 层"与"某个全注意力层"的名字清单，找映射规则。"""
import json, pathlib, struct
from collections import Counter

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF_DIR = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
objs = [o for o in man["objects"] if o.get("kind") == "tensor"]

def layer_of(nm):
    p = nm.split("/")
    if len(p) > 3 and p[0] == "text" and p[1] == "layers":
        return int(p[2])
    return None

by_layer = {}
for o in objs:
    L = layer_of(o["name"])
    by_layer.setdefault(L, []).append(o)

print("=== ninfer: 有哪些层号 ===")
layers = sorted(k for k in by_layer if k is not None)
print("  层数=%d  前 8=%s  后 4=%s" % (len(layers), layers[:8], layers[-4:]))
print("\n=== ninfer: 第 0 层的完整张量名 ===")
for o in sorted(by_layer.get(0, []), key=lambda x: x["name"]):
    print("   %-52s %-22s %s" % (o["name"], o["format"], o.get("shape")))
print("\n=== ninfer: 非层张量（顶层）===")
for o in objs:
    if layer_of(o["name"]) is None:
        print("   %-52s %-22s %s" % (o["name"], o["format"], o.get("shape")))

# HF 侧
idx = json.loads((HF_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = list(idx["weight_map"])
print("\n=== HF: 名字前缀分布 ===")
pref = Counter(".".join(k.split(".")[:4]) for k in wm)
for k, c in pref.most_common(12):
    print("   %-58s %d" % (k, c))
print("\n=== HF: 第 0 层全部名字 ===")
for k in sorted([x for x in wm if x.startswith("model.language_model.layers.0.")]):
    print("   %s" % k)
print("\n=== HF: 顶层（非 layer）名字 ===")
for k in sorted([x for x in wm if ".layers." not in x]):
    print("   %s" % k)
