#!/usr/bin/env python3
import json
import os
import struct
import numpy as np

ROOT = "/mnt/c/Users/User/Documents/ziqinzhang"
HF = os.path.join(ROOT, "models", "Qwen3.8-27B-NVFP4-RTX5090")
NF = os.path.join(ROOT, "models", "Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4",
                  "qwen3_8_27b_nvfp4.ninfer")


def e4m3_lut():
    lut = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0xF and m == 7:
            lut[b] = np.nan
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            lut[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return lut


L = e4m3_lut()

f = open(NF, "rb")
f.read(8)
n = int.from_bytes(f.read(8), "little")
hdr = json.loads(f.read(n).decode())
PO = ((16 + n + 4095) // 4096) * 4096
o = [x for x in hdr["objects"] if x["name"] == "text/token_embedding"][0]
N, K = o["shape"]

R = 64
f.seek(PO + o["offset"])
codes = np.frombuffer(f.read(R * K), np.uint8).reshape(R, K)
f.seek(PO + o["offset"] + N * K)
scales = np.frombuffer(f.read(R * 2), "<f2").astype(np.float32)
f.close()
B = L[codes] * scales[:, None]
print("ninfer token_embedding[0:%d]" % R)
print("  codes min/max      :", codes.min(), codes.max())
print("  scales  min/max/med: %.4g %.4g %.4g" % (scales.min(), scales.max(), np.median(scales)))
print("  B      absmax/std  : %.4g %.4g   row0 norm %.4g" %
      (np.abs(B).max(), B.std(), np.linalg.norm(B[0])))

# HF side
idx = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]
name = "model.language_model.embed_tokens.weight"
p = os.path.join(HF, idx[name])
with open(p, "rb") as fh:
    hl = 8 + struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(open(p, "rb").read(hl - 8)[8:].decode() if False else b"{}") if False else None
# read header properly
with open(p, "rb") as fh:
    ln = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(ln).decode())
e = h[name]
off = hl + e["data_offsets"][0]
with open(p, "rb") as fh:
    fh.seek(off)
    A = np.frombuffer(fh.read(R * K * 2), "<f2").reshape(R, K).astype(np.float32)
print("HF embed_tokens[0:%d]" % R)
print("  dtype/shape        :", e["dtype"], e["shape"], "abs_off", off)
print("  A      absmax/std  : %.4g %.4g   row0 norm %.4g" %
      (np.abs(A).max(), A.std(), np.linalg.norm(A[0])))

# row-norm ratio distribution
na = np.linalg.norm(A, axis=1)
nb = np.linalg.norm(B, axis=1)
r = nb / np.maximum(na, 1e-30)
print("  row-norm ratio B/A : min %.4f med %.4f max %.4f" %
      (r.min(), np.median(r), r.max()))

# cosine per row
cs = (A * B).sum(1) / np.maximum(na * nb, 1e-30)
print("  per-row cos        : min %.4f med %.4f max %.4f" %
      (cs.min(), np.median(cs), cs.max()))

# What if the scale convention is 1/scale ?
B2 = L[codes] / scales[:, None]
n2 = np.linalg.norm(B2, axis=1)
c2 = (A * B2).sum(1) / np.maximum(na * n2, 1e-30)
print("  [alt] codes/scale  : row-norm ratio med %.4f  per-row cos med %.4f"
      % (np.median(n2 / na), np.median(c2)))

# What if codes are column-major (transposed plane)?
B3 = L[codes.T[:K, :R]].T * scales[:, None] if False else None

# compare distributions of raw fp8 code values (scale-free)
print("  raw code value std : ninfer %.4g   HF(sign*|.|) %.4g"
      % (L[codes].std(), np.abs(A).std()))
