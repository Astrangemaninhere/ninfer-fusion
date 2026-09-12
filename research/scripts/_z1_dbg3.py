#!/usr/bin/env python3
"""Validate offset arithmetic on both sides using RMSNorm weights (expect ~O(1))."""
import json
import os
import struct
import numpy as np

ROOT = "/mnt/c/Users/User/Documents/ziqinzhang"
HF = os.path.join(ROOT, "models", "Qwen3.8-27B-NVFP4-RTX5090")
NF = os.path.join(ROOT, "models", "Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4",
                  "qwen3_8_27b_nvfp4.ninfer")

# ---- HF ----
idx = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]
p = os.path.join(HF, idx["model.language_model.layers.0.input_layernorm.weight"])
with open(p, "rb") as fh:
    ln = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(ln).decode())
hdr_len = 8 + ln
e = h["model.language_model.layers.0.input_layernorm.weight"]
off = hdr_len + e["data_offsets"][0]
with open(p, "rb") as fh:
    fh.seek(off)
    raw = fh.read(e["data_offsets"][1] - e["data_offsets"][0])
v = np.frombuffer(raw, "<f2").astype(np.float32)
print("HF layers.0.input_layernorm.weight  shape=%s off=%d  n=%d" % (e["shape"], off, v.size))
print("   mean %.6f std %.6f min %.4f max %.4f" % (v.mean(), v.std(), v.min(), v.max()))

# also check the very first tensor in shard 1 by data_offsets order
first = min(h.items(), key=lambda kv: kv[1]["data_offsets"][0] if "data_offsets" in kv[1] else 1 << 60)
print("   first tensor in shard1:", first[0], first[1]["dtype"], first[1]["shape"],
      first[1]["data_offsets"])
# file data size vs last offset
data_len = os.path.getsize(p) - hdr_len
mx = max(kv[1]["data_offsets"][1] for kv in h.values() if "data_offsets" in kv[1])
print("   data section: file=%d  max_store=%d  match=%s" % (data_len, mx, data_len >= mx))

# ---- ninfer ----
f = open(NF, "rb")
f.read(8)
n = int.from_bytes(f.read(8), "little")
hdr = json.loads(f.read(n).decode())
PO = ((16 + n + 4095) // 4096) * 4096
for nm in ("text/layers/0/input_norm", "text/final_norm", "text/layers/0/gdn/norm"):
    o = [x for x in hdr["objects"] if x["name"] == nm][0]
    f.seek(PO + o["offset"])
    b = f.read(o["bytes"])
    if o["format"] == "BF16":
        vv = np.frombuffer(b, "<f2").astype(np.float32)
    elif o["format"] == "FP32":
        vv = np.frombuffer(b, "<f4")
    else:
        vv = None
    print("NF %-28s %-8s %s -> %s" % (nm, o["format"], o["shape"],
                                       ("mean %.5f std %.5f min %.4f max %.4f"
                                        % (vv.mean(), vv.std(), vv.min(), vv.max()))
                                       if vv is not None else "n/a"))
f.close()
