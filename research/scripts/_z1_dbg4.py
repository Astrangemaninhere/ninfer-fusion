#!/usr/bin/env python3
import json
import os
import struct
import numpy as np

ROOT = "/mnt/c/Users/User/Documents/ziqinzhang"
HF = os.path.join(ROOT, "models", "Qwen3.8-27B-NVFP4-RTX5090")
p = os.path.join(HF, "model-00001-of-00002.safetensors")

with open(p, "rb") as fh:
    ln = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(ln).decode())
hdr_len = 8 + ln
size = os.path.getsize(p)
print("file size = %d   header_len = %d   data section = %d" % (size, ln, size - hdr_len))
print("metadata key present:", "__metadata__" in h)

ents = [(v["data_offsets"][0], v["data_offsets"][1], k, v.get("dtype"), tuple(v.get("shape", [])))
        for k, v in h.items() if k != "__metadata__"]
ents.sort()
print("n tensors =", len(ents))
print("\nfirst 6 by offset:")
for a, b, k, dt, sh in ents[:6]:
    print("   [%12d,%12d) %-8s %-16s %s" % (a, b, dt, str(sh), k))
print("\ncoverage check:")
mx = max(b for _, b, _, _, _ in ents)
print("   max end = %d   data section = %d   match=%s" % (mx, size - hdr_len, mx == size - hdr_len))
gaps = 0
cur = 0
for a, b, _, _, _ in ents:
    if a != cur:
        gaps += 1
    cur = b
print("   contiguous from 0:", gaps == 0, " gaps:", gaps)

# read the *first* tensor and print a few values
a, b, k, dt, sh = ents[0]
with open(p, "rb") as fh:
    fh.seek(hdr_len + a)
    raw = fh.read(b - a)
npdt = {"BF16": "<f2", "F32": "<f4", "F16": "<f2", "U8": "u1", "I8": "i1",
        "F8_E4M3": "u1", "F8_E4M3FN": "u1"}.get(dt)
if npdt:
    v = np.frombuffer(raw, npdt)
    print("\nfirst tensor %s (%s %s): raw[:8]=%s  size=%d" % (k, dt, sh, v[:8].tolist(), v.size))

# try an alternate interpretation: data_offsets relative to 0 of file (no hdr) -- print both
for name in ("model.language_model.embed_tokens.weight",
             "model.language_model.layers.0.input_layernorm.weight",
             "model.language_model.norm.weight"):
    if name not in h:
        print("MISSING", name)
        continue
    e = h[name]
    print("\n%s\n   dtype=%s shape=%s data_offsets=%s"
          % (name, e["dtype"], e["shape"], e["data_offsets"]))
    o = hdr_len + e["data_offsets"][0]
    with open(p, "rb") as fh:
        fh.seek(o)
        raw = fh.read(min(e["data_offsets"][1] - e["data_offsets"][0], 4096))
    v = np.frombuffer(raw, "<f2").astype(np.float32)
    print("   abs_off=%d  first8=%s  mean=%.4f std=%.4f" % (o, v[:8].tolist(), v.mean(), v.std()))
