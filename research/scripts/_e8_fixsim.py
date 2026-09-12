#!/usr/bin/env python3
"""Quantify the §116 fix BEFORE touching code: compare the e8 K reconstruction
error with (A) the shipped scale = pre-rotation max/7, and (B) the proposed
scale = post-rotation max/7. Both use the same E8 projection as the kernel."""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_e8src"


def bf16(u16):
    return ((u16.astype(np.uint32)) << 16).view(np.float32)


def hadamard64(x):
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


def e8_project_8(x):
    """Nearest point of E8 (D8 u (D8+0.5)) per 8-dim block, like the kernel."""
    v = x.reshape(-1, 8).astype(np.float32)
    f = np.rint(v)
    s = f.sum(axis=1, keepdims=True)
    d8 = f.copy()
    # parity fix on the coordinate with the largest rounding error
    err = np.abs(v - f)
    worst = err.argmax(axis=1)
    rows = np.arange(v.shape[0])
    need = (s[:, 0].astype(np.int64) & 1) != 0
    d8[rows[need], worst[need]] += np.where(v[rows[need], worst[need]] >= f[rows[need], worst[need]],
                                            1.0, -1.0)
    xs = v - 0.5
    fs = np.rint(xs)
    ss = fs.sum(axis=1, keepdims=True)
    c1 = fs + 0.5
    errs = np.abs(xs - fs)
    worsts = errs.argmax(axis=1)
    needs = (ss[:, 0].astype(np.int64) & 1) != 0
    c1[rows[needs], worsts[needs]] += np.where(
        xs[rows[needs], worsts[needs]] >= fs[rows[needs], worsts[needs]], 1.0, -1.0)
    take0 = ((d8 - v) ** 2).sum(axis=1) <= ((c1 - v) ** 2).sum(axis=1)
    return np.where(take0[:, None], d8, c1).reshape(x.shape)


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
    meta = read_meta(metas[0])
    head = meta.get("layer=%d" % layer) or meta.get("layer", {})
    head_dim = int(head.get("head_dim", 256))
    kv_heads = int(head.get("num_kv_heads", 4))
    src = bf16(np.frombuffer(open(srcs[0], "rb").read(), dtype="<u2")).reshape(
        head_dim, kv_heads, -1)
    err_a, err_b = [], []
    clamp_a, clamp_b = [], []
    for t in range(min(64, src.shape[2])):
        for h in range(kv_heads):
            for g0 in range(0, head_dim, 64):
                block = src[g0:g0 + 64, h, t].astype(np.float32)
                rot = hadamard64(block)
                pre_max = float(np.abs(block).max())
                post_max = float(np.abs(rot).max())
                for tag, mx, store in (("A", pre_max, err_a), ("B", post_max, err_b)):
                    s = mx / 7.0 if mx > 0 else 0.0
                    q = e8_project_8(rot / s) if s > 0 else np.zeros_like(rot)
                    q = np.clip(q, -7.0, 7.0)
                    recon = q * s
                    store.append(float(np.abs(recon - rot).mean() / (np.abs(rot).mean() + 1e-9)))
                    (clamp_a if tag == "A" else clamp_b).append(float((np.abs(q) >= 7.0).mean()))
    print("L%d: rel-err A(pre-rot)/B(post-rot) = %.4f / %.4f  (median per-group) | "
          "clamp-rate A/B = %.3f / %.3f"
          % (layer, float(np.median(err_a)), float(np.median(err_b)),
             float(np.mean(clamp_a)), float(np.mean(clamp_b))))
