#!/usr/bin/env python3
"""§96 numerical probe: relative error of the stored KV vs the append source,
per layer and per position (e8 vs nvfp4 tiers).

Needs a dump that contains BOTH:
  kvsrc_<id>_L<layer>_kn.bin            append-source K (BF16, {head_dim, kv_heads, T})
  kvc_<id>_t<T>_L<layer>_{k,ks,meta}    stored planes (+ meta with ne/dtype)
Usage: _e8_error.py <dumpdir> [layers]
"""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_e8src"
WANT = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "13,14,15".split(","))]

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e4m3(b):
    b = np.asarray(b, dtype=np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    val = np.where(exp == 0, mant * 2.0 ** -9, (1.0 + mant / 8.0) * np.power(2.0, exp - 7))
    return sign * np.where((exp == 15) & (mant == 7), np.nan, val)


def bf16(u16):
    return ((u16.astype(np.uint32)) << 16).view(np.float32)


def hadamard8(x):
    """Replicates gqa_kv_hadamard64 (gqa_attention_kv_quant.cuh): a 64-dim
    Sylvester butterfly where lanes hold dims (d, d+32) of one 64-group.
    x: (64, C) -> (64, C), orthonormal (entries +-1/8)."""
    x0 = x[:32].astype(np.float32).copy()
    x1 = x[32:].astype(np.float32).copy()
    lanes = np.arange(32)
    for offset in (1, 2, 4, 8, 16):
        idx = lanes ^ offset
        y0 = x0[idx]
        y1 = x1[idx]
        hi = ((lanes & offset) != 0)[:, None]
        x0 = np.where(hi, y0 - x0, x0 + y0)
        x1 = np.where(hi, y1 - x1, x1 + y1)
    a = x0.copy()
    b = x1.copy()
    x0 = (a + b) * 0.125
    x1 = (a - b) * 0.125
    return np.concatenate([x0, x1], axis=0)


def read_meta(path):
    m = {}
    for line in open(path):
        p = line.split()
        if p:
            m[p[0]] = dict(x.split("=", 1) for x in p[1:] if "=" in x)
    return m


def main():
    for layer in WANT:
        srcs = sorted(glob.glob(os.path.join(DUMP, "kvsrc_*_L%d_kn.bin" % layer)))
        planes = sorted(glob.glob(os.path.join(DUMP, "kvc_*_L%d_meta.txt" % layer)))
        if not srcs or not planes:
            print("L%d: missing src (%d) or planes (%d)" % (layer, len(srcs), len(planes)))
            continue
        meta = read_meta(planes[0])
        head = meta.get("layer=%d" % layer) or meta.get("layer", {})
        view_dtype = int(head.get("dtype", 0))
        head_dim = int(head.get("head_dim", 256))
        kv_heads = int(head.get("num_kv_heads", 4))
        group = int(head.get("quant_group", 64)) or 64
        k_shape = [int(x) for x in meta["k"]["ne"].split(",")]
        ks_shape = [int(x) for x in meta["ks"]["ne"].split(",")]
        pre = planes[0][: -len("_meta.txt")]
        k_raw = np.fromfile(pre + "_k.bin", dtype=np.uint8)
        ks_raw = np.fromfile(pre + "_ks.bin", dtype=np.uint8)
        src = bf16(np.fromfile(srcs[0], dtype="<u2")).reshape(head_dim, kv_heads, -1)
        T = src.shape[2]
        pages = k_raw.size // (k_shape[0] * k_shape[1] * k_shape[2])
        k_arr = k_raw.reshape(k_shape[0], k_shape[1], k_shape[2], pages)
        tier = "e8" if view_dtype == 10 else ("nvfp4" if view_dtype == 8 else "other(%d)" % view_dtype)
        print("=== L%d tier=%s head_dim=%d kv_heads=%d group=%d T=%d k_ne=%s ks_ne=%s"
              % (layer, tier, head_dim, kv_heads, group, T, k_shape, ks_shape))
        rel = []
        for t in range(min(T, 64)):
            codes = k_arr[:, t, :, 0]                     # (head_dim//2, kv_heads) u8
            q = np.zeros((head_dim, kv_heads), dtype=np.float32)
            if tier == "e8":
                scales = np.frombuffer(ks_raw, dtype="<f2").reshape(
                    ks_shape[0], ks_shape[1], ks_shape[2], pages)[:, t, :, 0]
                gscale = np.repeat(scales.astype(np.float32), group, axis=0)
                lo = (codes & 0x0F).astype(np.int16)
                hi = ((codes >> 4) & 0x0F).astype(np.int16)
                lo = np.where(lo > 7, lo - 16, lo)
                hi = np.where(hi > 7, hi - 16, hi)
                q[0::2] = lo
                q[1::2] = hi
                recon = q * gscale
                s = src[:, :, t]
                ref = np.empty_like(s)
                for g0 in range(0, head_dim, 64):
                    ref[g0:g0 + 64] = hadamard8(s[g0:g0 + 64])
            else:
                scales = e4m3(np.frombuffer(ks_raw, dtype=np.uint8).reshape(
                    ks_shape[0], ks_shape[1], ks_shape[2], pages)[:, t, :, 0])
                gscale = np.repeat(scales.astype(np.float32), group, axis=0)
                q[0::2] = E2M1[codes & 0x0F]
                q[1::2] = E2M1[(codes >> 4) & 0x0F]
                recon = q * gscale
                ref = src[:, :, t]
            rel.append(float(np.abs(recon - ref).mean() / (np.abs(ref).mean() + 1e-9)))
        if rel:
            print("    rel-err mean=%.4f  first8=%s  mid8=%s  last8=%s"
                  % (float(np.mean(rel)), np.round(rel[:8], 4).tolist(),
                     np.round(rel[len(rel) // 2:len(rel) // 2 + 8], 4).tolist(),
                     np.round(rel[-8:], 4).tolist()))


if __name__ == "__main__":
    main()
