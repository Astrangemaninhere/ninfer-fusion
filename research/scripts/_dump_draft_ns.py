#!/usr/bin/env python3
"""列 dflash2 / dspark 产物里草稿层（dflash2/ dflash/ 命名空间）的全部张量，
看是否存在"独立的 context_key/context_value"以及它们与 qkv 的关系。"""
import json, struct

def load(path):
    with open(path, "rb") as f:
        f.read(8); n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace"))

for tag, path in (("dflash2", "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"),
                  ("dspark",  "/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer")):
    man = load(path)
    tensors = [o for o in man["objects"] if o.get("kind") == "tensor"]
    ns = {}
    for o in tensors:
        top = o["name"].split("/")[0]
        ns.setdefault(top, []).append(o)
    print("=== %s ===  identity=%s  总张量=%d" % (tag, json.dumps(man.get("identity"), ensure_ascii=False), len(tensors)))
    print("   命名空间: %s" % {k: len(v) for k, v in sorted(ns.items())})
    key = "dflash2" if "dflash2" in ns else ("dflash" if "dflash" in ns else None)
    if key:
        print("   --- %s/ 下全部张量 ---" % key)
        for o in sorted(ns[key], key=lambda x: x["name"]):
            print("     %-58s %-22s %s" % (o["name"], o["format"], o.get("shape")))
    # 找 context / kv 相关
    print("   --- 含 context/kv/key/value 的名字（全命名空间）---")
    for o in tensors:
        nm = o["name"].lower()
        if any(w in nm for w in ("context", "kv", "key", "value", "qkv")):
            print("     %-58s %-22s %s" % (o["name"], o["format"], o.get("shape")))
    print()
