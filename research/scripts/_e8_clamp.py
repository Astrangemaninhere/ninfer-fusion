#!/usr/bin/env python3
"""Test the "scale taken from the PRE-rotation max" hypothesis for the e8 tier:
if true, post-rotation values exceed 7*scale and the codes clamp at +-7."""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_e8src"


def bf16(u16):
    return ((u16.astype(np.uint32)) << 16).view(np.float32)


def hadamard8(x):
    x0 = x[:32].astype(np.float32).copy()
    x1 = x[32:].astype(np.float32).copy()
    lanes = np.arange(32)
    for offset in (1, 2, 4, 8, 16):
        idx = lanes ^ offset
        y0, y1 = x0[idx], x1[idx]
        hi = ((lanes & offset) != 0)[:, None]
        x0 = np.where(hi, y0 - x0, x0 + y0)
        x1 = np.where(hi, y1 - x1, x1 + y1)
    a, b = x0.copy(), x1.copy()
    return np.concatenate([(a + b) * 0.125, (a - b) * 0.125], axis=0)


def read_meta(path):
    m = {}
    for line in open(path):
        p = line.split()
        if p:
            m[p[0]] = dict(x.split("=", 1) for x in p[1:] if "=" in x)
    return m


for layer in (14, 15):
    srcs = sorted(glob.glob(os.path.join(DUMP, "kvsrc_*_L%d_kn.bin" % layer)))
    metas = sorted(glob.glob(os.path.join(DUMP, "kvc_*_L%d_meta.txt" % layer)))
    if not srcs or not metas:
        print("L%d: missing dumps" % layer)
        continue
    pre = metas[0][: -len("_meta.txt")]
    meta = read_meta(metas[0])
    head = meta.get("layer=%d" % layer) or meta.get("layer", {})
    head_dim = int(head.get("head_dim", 256))
    kv_heads = int(head.get("num_kv_heads", 4))
    src = bf16(np.fromfile(srcs[0], dtype="<u2")).reshape(head_dim, kv_heads, -1)
    k_raw = np.fromfile(pre + "_k.bin", dtype=np.uint8)
    ks_raw = np.fromfile(pre + "_ks.bin", dtype=np.uint8)
    k_shape = [int(x) for x in meta["k"]["ne"].split(",")]
    ks_shape = [int(x) for x in meta["ks"]["ne"].split(",")]
    pages = k_raw.size // (k_shape[0] * k_shape[1] * k_shape[2])
    k_arr = k_raw.reshape(k_shape[0], k_shape[1], k_shape[2], pages)
    scales = np.frombuffer(ks_raw, dtype="<f2").reshape(ks_shape[0], ks_shape[1], ks_shape[2], pages)
    n_code = 0
    n_clamp = 0
    over = []
    for t in range(min(64, src.shape[2])):
        for h in range(kv_heads):
            codes = k_arr[:, t, h, 0]
            lo = (codes & 0x0F).astype(np.int16)
            hi = ((codes >> 4) & 0x0F).astype(np.int16)
            lo = np.where(lo > 7, lo - 16, lo)
            hi = np.where(hi > 7, hi - 16, hi)
            allc = np.concatenate([lo, hi])
            n_code += allc.size
            n_clamp += int((np.abs(allc) == 7).sum())
            for g in range(head_dim // 64):
                s = float(scales[g, t, h, 0])
                block = src[g * 64:(g + 1) * 64, h, t]
                rot = hadamard8(block)
                # pre-rotation max (what the kernel scales with) vs post-rotation max
                pre_max = float(np.abs(block).max())
                post_max = float(np.abs(rot).max())
                if s > 0:
                    over.append(post_max / (7.0 * s))
    print("L%d: codes=%d  |code|==7 %d (%.4f)  post/pre max ratio median=%.3f max=%.3f"
          % (layer, n_code, n_clamp, n_clamp / n_code,
             float(np.median(over)) if over else -1, float(np.max(over)) if over else -1))
