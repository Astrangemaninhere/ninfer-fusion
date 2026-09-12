#!/usr/bin/env python3
"""Compare stored K/V against the append-source K/V for the same layer, and
cross-compare sources across layers (are the sources themselves identical?)."""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_src"
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


def read_meta(path):
    m = {}
    for line in open(path):
        p = line.split()
        if p:
            m[p[0]] = dict(x.split("=", 1) for x in p[1:] if "=" in x)
    return m


def decode_k(codes, scales, head_dim, group):
    L, H = codes.shape
    out = np.zeros((head_dim, H), dtype=np.float32)
    for i in range(L):
        s = e4m3(scales[(2 * i) // group])
        out[2 * i] = E2M1[codes[i] & 0x0F] * s
        out[2 * i + 1] = E2M1[(codes[i] >> 4) & 0x0F] * s
    return out


srcs = {}
for p in sorted(glob.glob(os.path.join(DUMP, "kvsrc_*_meta.txt"))):
    base = p[: -len("_meta.txt")]
    layer = int(os.path.basename(base).split("_L")[1])
    m = read_meta(p)
    kn_shape = [int(x) for x in m["kn"]["ne"].split(",")]
    kn = bf16(np.fromfile(base + "_kn.bin", dtype="<u2").reshape(kn_shape[0], kn_shape[1], -1))
    srcs[layer] = kn
    print("src L%d kn shape=%s md5[:8]=%s" % (layer, kn.shape,
                                              __import__("hashlib").md5(kn.tobytes()).hexdigest()[:8]))
layers = sorted(srcs)
if len(layers) > 1:
    a = srcs[layers[0]]
    for l in layers[1:]:
        b = srcs[l]
        n = min(a.size, b.size)
        print("  src L%d vs L%d identical=%s maxdiff=%.4g"
              % (layers[0], l, np.array_equal(a[:, :, :n // a.shape[0] // a.shape[1]],
                                              b[:, :, :n // a.shape[0] // a.shape[1]]),
                 np.abs(a[:, :, :n // a.shape[0] // a.shape[1]] -
                        b[:, :, :n // a.shape[0] // a.shape[1]]).max()))

for meta_path in sorted(glob.glob(os.path.join(DUMP, "kvc_0_*_L*_meta.txt"))):
    pre = meta_path[: -len("_meta.txt")]
    layer = int(os.path.basename(pre).split("_L")[1])
    m = read_meta(meta_path)
    k_shape = [int(x) for x in m["k"]["ne"].split(",")]
    ks_shape = [int(x) for x in m["ks"]["ne"].split(",")]
    k_raw = np.fromfile(pre + "_k.bin", dtype=np.uint8)
    ks_raw = np.fromfile(pre + "_ks.bin", dtype=np.uint8)
    pages = k_raw.size // (k_shape[0] * k_shape[1] * k_shape[2])
    k_arr = k_raw.reshape(k_shape[0], k_shape[1], k_shape[2], pages)
    ks_arr = ks_raw.reshape(ks_shape[0], ks_shape[1], ks_shape[2], pages)
    if layer not in srcs:
        continue
    kn = srcs[layer]
    head_dim = kn.shape[0]
    print("--- L%d stored vs source (page 0)" % layer)
    for t in (0, 1, 2):
        dec = decode_k(k_arr[:, t, :, 0], ks_arr[:, t, :, 0], head_dim, 16)
        ref = kn[:, :, t]
        print("    tok %d: stored max=%.4g  src max=%.4g  maxdiff=%.4g  corr=%.4f"
              % (t, np.abs(dec).max(), np.abs(ref).max(), np.abs(dec - ref).max(),
                 np.corrcoef(dec.ravel(), ref.ravel())[0, 1]))
