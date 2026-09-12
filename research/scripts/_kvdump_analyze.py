#!/usr/bin/env python3
"""Decode a KV-dump (NINFER_KVDUMP_DIR) and compare stored K/V against the
source BF16 K/V the append path consumed.

Engine probes (text_context_impl.h / text_prefill_impl.h):
  kvc_<id>_t<T>_bt.bin / _bt_meta.txt      block tables (i32 {logical_pages, rows})
  kvc_<id>_t<T>_L<L>_{k,v,ks,vs}.bin       raw planes {lead, 64, kv_heads, pages}
  kvc_<id>_t<T>_L<L>_meta.txt              plane shapes / dtypes
  kvsrc_<id>_L<L>_{kn,v,pos}.bin           source BF16 K/V + positions (i32)
  kvsrc_<id>_L<L>_meta.txt
"""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump"
MAX_TOKENS = int(os.environ.get("KVDUMP_TOKENS", "8"))

# e2m1 (nvfp4 code) -> float32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e4m3_to_f32(b):
    b = np.asarray(b, dtype=np.uint16)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    val = np.where(exp == 0, mant * 2.0 ** -9, (1.0 + mant / 8.0) * np.power(2.0, exp - 7))
    val = np.where((exp == 15) & (mant == 7), np.nan, val)
    return sign * val


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            meta[parts[0]] = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    return meta


def fld(meta, name, key, default="0"):
    return meta.get(name, {}).get(key, default)


def shape_of(meta, name):
    return [int(x) for x in fld(meta, name, "ne").split(",")]


def load_plane(prefix, meta, name):
    shape = shape_of(meta, name)
    if not shape or shape[0] == 0:
        return None, None
    path = prefix + "_" + name + ".bin"
    if not os.path.exists(path):
        return None, None
    itemsize = 2 if name in ("ks", "vs") and fld(meta, name, "dtype") in ("1", "2") else 1
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(np.prod(shape))
    if raw.size < n * itemsize:
        print("  ! short read %s (%d < %d)" % (path, raw.size, n * itemsize))
        return None, shape
    return raw[: n * itemsize], shape


def bf16_to_f32(arr_u16):
    return ((arr_u16.astype(np.uint32)) << 16).view(np.float32)


def decode_k_nvfp4(codes, scales, head_dim, group):
    """codes (head_dim//2, H) u8, scales (head_dim//group, H) u8 -> (head_dim, H) f32."""
    L, H = codes.shape
    out = np.zeros((head_dim, H), dtype=np.float32)
    for i in range(L):
        s = e4m3_to_f32(scales[(2 * i) // group])
        out[2 * i] = E2M1[codes[i] & 0x0F] * s
        out[2 * i + 1] = E2M1[(codes[i] >> 4) & 0x0F] * s
    return out


def main():
    bt_metas = sorted(glob.glob(os.path.join(DUMP, "kvc_*_bt_meta.txt")))
    if not bt_metas:
        print("no dumps in", DUMP)
        return
    for bt_meta_path in bt_metas:
        base = bt_meta_path[: -len("_bt_meta.txt")]
        bt_meta = read_meta(bt_meta_path)
        bt_shape = shape_of(bt_meta, "block_tables")
        bt = np.fromfile(base + "_bt.bin", dtype="<i4")
        if bt.size < bt_shape[0] * bt_shape[1]:
            print("bt short read", base)
            continue
        bt = bt.reshape(bt_shape[1], bt_shape[0])
        print("=== %s tokens=%s bt=%s" % (os.path.basename(base), fld(bt_meta, "tokens", "tokens"), bt_shape))
        rows = [r for r in range(bt.shape[0]) if (bt[r] >= 0).any()]
        for r in rows:
            print("  row %d pages[:6]=%s" % (r, bt[r][:6].tolist()))
        layers = sorted(set(int(os.path.basename(p).split("_L")[1].split("_")[0])
                            for p in glob.glob(base + "_L*_meta.txt")))
        for layer in layers:
            prefix = "%s_L%d" % (base, layer)
            meta = read_meta(prefix + "_meta.txt")
            head = meta.get("layer", {})
            head_dim = int(head.get("head_dim", 0))
            kv_heads = int(head.get("num_kv_heads", 0))
            dtype = int(head.get("dtype", 1))
            group = int(head.get("quant_group", 0)) or 16
            k_raw, k_shape = load_plane(prefix, meta, "k")
            v_raw, v_shape = load_plane(prefix, meta, "v")
            ks_raw, ks_shape = load_plane(prefix, meta, "ks")
            vs_raw, vs_shape = load_plane(prefix, meta, "vs")
            print("--- L%d dtype=%d head_dim=%d kv_heads=%d group=%d k_shape=%s ks_shape=%s"
                  % (layer, dtype, head_dim, kv_heads, group, k_shape, ks_shape))
            if ks_raw is not None:
                nan_mask = ((ks_raw & 0x7F) > 0x7E) & (ks_raw != 0x80)
                print("    ks: nan_bytes=%d ff=%d zero=%d of %d"
                      % (nan_mask.sum(), (ks_raw == 0xFF).sum(), (ks_raw == 0).sum(), ks_raw.size))
            if k_raw is not None:
                print("    k : ff=%d zero=%d of %d"
                      % ((k_raw == 0xFF).sum(), (k_raw == 0).sum(), k_raw.size))
            # source K/V
            src = None
            src_meta = None
            for cand in sorted(glob.glob(os.path.join(DUMP, "kvsrc_*_L%d_meta.txt" % layer))):
                m = read_meta(cand)
                if "kn" in m:
                    src = cand[: -len("_meta.txt")]
                    src_meta = m
                    break
            if src is None:
                print("    (no source dump for this layer)")
                continue
            kn_shape = shape_of(src_meta, "kn")
            v_shape_src = shape_of(src_meta, "v")
            kn = bf16_to_f32(np.fromfile(src + "_kn.bin", dtype="<u2").reshape(
                kn_shape[0], kn_shape[1], -1))
            vv = bf16_to_f32(np.fromfile(src + "_v.bin", dtype="<u2").reshape(
                v_shape_src[0], v_shape_src[1], -1))
            pos = np.fromfile(src + "_pos.bin", dtype="<i4")
            T = kn.shape[2]
            print("    src kn nan=%d inf=%d maxabs=%.4g | v nan=%d inf=%d maxabs=%.4g | pos[:6]=%s"
                  % (np.isnan(kn).sum(), np.isinf(kn).sum(), np.nanmax(np.abs(kn)),
                     np.isnan(vv).sum(), np.isinf(vv).sum(), np.nanmax(np.abs(vv)),
                     pos[:6].tolist()))
            if k_raw is None or dtype != 7:
                for t in range(min(MAX_TOKENS, T)):
                    col = kn[:, :, t]
                    print("      src tok %2d: maxabs=%.4g nan=%d" % (t, np.nanmax(np.abs(col)), np.isnan(col).sum()))
                continue
            # nvfp4 stored-vs-source comparison for logical page 0 of each used row
            L, tokens_per_page, H, pages = k_shape
            k_arr = k_raw.reshape(L, tokens_per_page, H, pages)
            ks_arr = ks_raw.reshape(ks_shape[0], ks_shape[1], ks_shape[2], ks_shape[3])
            page0 = int(bt[rows[0]][0]) if rows else 0
            if page0 < 0:
                print("    logical page 0 unmapped")
                continue
            print("    stored vs source (page %d, first %d tokens):" % (page0, min(MAX_TOKENS, T)))
            for t in range(min(MAX_TOKENS, T)):
                codes = k_arr[:, t, :, page0]
                scales = ks_arr[:, t, :, page0]
                dec = decode_k_nvfp4(codes, scales, head_dim, group)
                ref = kn[:, :, t]
                diff = np.abs(dec - ref)
                print("      tok %2d: stored maxabs=%.4g nan=%d | src maxabs=%.4g | maxdiff=%.4g "
                      "scale_nan=%d" % (t, np.nanmax(np.abs(dec)), np.isnan(dec).sum(),
                                        np.nanmax(np.abs(ref)), np.nanmax(diff),
                                        np.isnan(e4m3_to_f32(scales)).sum()))


if __name__ == "__main__":
    main()
