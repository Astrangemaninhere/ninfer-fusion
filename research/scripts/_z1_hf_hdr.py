#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z1: dump the HF target's safetensors headers for the tensors we want to
compare against the .ninfer artifact."""
import json
import os
import struct

T = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090"
idx = json.load(open(T + "/model.safetensors.index.json"))["weight_map"]


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


cache = {}
WANT = [
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.3.mlp.down_proj.weight",
    "model.language_model.layers.3.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.3.linear_attn.out_proj.weight",
]
for name in WANT:
    shard = idx.get(name)
    if shard is None:
        print("%-58s  <NOT IN INDEX>" % name)
        continue
    p = os.path.join(T, shard)
    if p not in cache:
        cache[p] = header(p)
    h = cache[p]
    print("== %s  (%s)" % (name, shard))
    stem = name
    for suf in ("", ".weight_scale", ".weight_scale_2", ".input_scale",
                ".input_scale_2"):
        k = stem + suf
        if k in h:
            e = h[k]
            print("   %-62s %-8s %s" % (k, e["dtype"], e["shape"]))
    print()
