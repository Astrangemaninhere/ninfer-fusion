#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z1: read the .ninfer header JSON (no ninfer code needed) and compare its
structure with the gittensor HF target's config / tensor index."""
import json
import re
import sys
import collections

NINFER = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4/qwen3_8_27b_nvfp4.ninfer"
HFIDX = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090/model.safetensors.index.json"
HFCONF = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090/config.json"

with open(NINFER, "rb") as f:
    magic = f.read(8)
    n = int.from_bytes(f.read(8), "little")
    hdr = f.read(n)
print("magic        :", magic)
print("header bytes :", n)
j = json.loads(hdr.decode("utf-8"))
print("identity     :", json.dumps(j["identity"]))
objs = j["objects"]
print("objects      :", len(objs))
kinds = collections.Counter(o["kind"] for o in objs)
print("kinds        :", dict(kinds))
fmts = collections.Counter(o.get("format") for o in objs if o["kind"] == "tensor")
print("tensor fmts  :", dict(fmts))

print("\n--- frontend resources ---")
for o in objs:
    if o["kind"] == "resource":
        print("   %-46s %s" % (o["name"], o.get("encoding")))

names = [o["name"] for o in objs if o["kind"] == "tensor"]
lay = collections.Counter()
for nm in names:
    m = re.match(r"text/layers/(\d+)/", nm)
    if m:
        lay[int(m.group(1))] += 1
ids = sorted(lay)
print("\n--- text layers ---")
print("count        :", len(ids), " range:", ids[0], "..", ids[-1])
miss = [i for i in range(ids[-1] + 1) if i not in lay]
print("missing      :", miss)

# layer type distribution: presence of gdn/ vs attention/
gdn = set()
attn = set()
for nm in names:
    m = re.match(r"text/layers/(\d+)/(\w+)", nm)
    if m:
        i, fam = int(m.group(1)), m.group(2)
        if fam == "gdn":
            gdn.add(i)
        elif fam not in ("input_norm", "post_attention_norm", "mlp", "norm"):
            attn.add(i)
print("layers with gdn/*        :", sorted(gdn)[:8], "... n=", len(gdn))
print("layers with other fams   :", sorted(attn)[:8], "... n=", len(attn))
fams = collections.Counter()
for nm in names:
    m = re.match(r"text/layers/\d+/([^/]+)", nm)
    if m:
        fams[m.group(1)] += 1
print("layer child families     :", dict(fams))

print("\n--- tensors of layer 0 and layer 3 (ninfer) ---")
for o in objs:
    if o["kind"] == "tensor" and re.match(r"text/layers/(0|3)/", o["name"]):
        print("   %-58s %-22s %s" % (o["name"], o["format"],
                                      "[" + ",".join(map(str, o["shape"])) + "]"))

print("\n--- text globals (ninfer) ---")
for o in objs:
    if o["kind"] == "tensor" and o["name"].startswith("text/") and "/layers/" not in o["name"]:
        print("   %-58s %-22s %s" % (o["name"], o["format"],
                                      "[" + ",".join(map(str, o["shape"])) + "]"))

print("\n--- HF target config ---")
c = json.load(open(HFCONF))
t = c["text_config"]
lt = t.get("layer_types") or []
print("vocab_size        :", t.get("vocab_size"))
print("hidden_size       :", t.get("hidden_size"))
print("num_hidden_layers :", t.get("num_hidden_layers"))
print("layer_types full_attention at:", [i for i, x in enumerate(lt) if x == "full_attention"])
print("tie_word_embeddings:", t.get("tie_word_embeddings"), " architectures:", c.get("architectures"))

wm = json.load(open(HFIDX))["weight_map"]
hf_attn = set()
hf_gdn = set()
for k in wm:
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(\w+)", k)
    if m:
        i, fam = int(m.group(1)), m.group(2)
        if fam == "linear_attn":
            hf_gdn.add(i)
        elif fam == "self_attn":
            hf_attn.add(i)
print("HF layers with linear_attn (GDN) at:", sorted(hf_gdn))
print("HF layers with self_attn   at:", sorted(hf_attn))
print("\nGDN layer sets identical (ninfer vs HF):", gdn == hf_gdn)
