#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z1: numerical same-source check, HF target vs the local .ninfer artifact.

Compares a few tensors of
  HF : models/Qwen3.8-27B-NVFP4-RTX5090  (gittensor, modelopt/compressed-tensors)
  NF : models/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4/qwen3_8_27b_nvfp4.ninfer
after dequantization, on a row slice.

Discriminators:
  cos            : cosine of the two dequantized slices
  relL2          : ||A-B|| / ||A||
  top1share      : sigma_1^2 / sum(sigma_i^2) of D = A - B
                   ~0.5%  => D is broadband quantisation noise  => same weights
                   >>10%  => D is low-rank (abliteration delta) => different weights
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = "/mnt/c/Users/User/Documents/ziqinzhang"
HF = os.path.join(ROOT, "models", "Qwen3.8-27B-NVFP4-RTX5090")
NF = os.path.join(ROOT, "models", "Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4",
                  "qwen3_8_27b_nvfp4.ninfer")
ROWS = 256

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3fn_lut():
    """256-entry LUT for OCP FP8 E4M3FN (bias 7, 1/4/3, 0x7F/0xFF = NaN)."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0x0F
        m = b & 0x07
        if e == 0x0F and m == 0x07:
            lut[b] = np.nan
        elif e == 0:
            lut[b] = s * (m / 8.0) * (2.0 ** -6)
        else:
            lut[b] = s * (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return lut


E4M3 = _e4m3fn_lut()


def fp8_e4m3fn_to_f32(u8):
    return E4M3[u8]


def nf_header():
    """Returns (header_json, payload_offset).

    Object offsets in the header are payload-relative; payload_offset itself is
    align_up(16 + len(header_json), 4096) -- container.py:18 PAYLOAD_ALIGNMENT.
    """
    with open(NF, "rb") as f:
        f.read(8)
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n).decode("utf-8"))
    meta_end = 16 + n
    A = 4096
    return hdr, ((meta_end + A - 1) // A) * A


PAYLOAD = 0


def nf_obj(hdr, name):
    for o in hdr["objects"]:
        if o["name"] == name:
            return o
    return None


def nf_read(off, nbytes):
    with open(NF, "rb") as f:
        f.seek(PAYLOAD + off)
        return f.read(nbytes)


def nf_dequant_fp8_row(name, shape, rows):
    """FP8_E4M3FN_ROW_BF16S: codes [n,k] u8 then n bf16 row scales."""
    n, k = shape
    o = nf_obj(H, name)
    codes = np.frombuffer(nf_read(o["offset"], rows * k), dtype=np.uint8).reshape(rows, k)
    scales = np.frombuffer(nf_read(o["offset"] + n * k, rows * 2), dtype="<f2")
    a = fp8_e4m3fn_to_f32(codes) * scales.astype(np.float32)[:, None]
    return a


def hf_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def hf_read(path, off, nbytes):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(nbytes)


def hf_tensor_header(shard, name):
    p = os.path.join(HF, shard)
    h = hf_safetensors_header(p)
    e = h[name]
    with open(p, "rb") as f:
        hdr_len = 8 + struct.unpack("<Q", f.read(8))[0]
    return e, hdr_len + e["data_offsets"][0]


def metrics(A, B, label):
    A = A.ravel().astype(np.float64)
    B = B.ravel().astype(np.float64)
    cos = float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B)))
    rel = float(np.linalg.norm(A - B) / np.linalg.norm(A))
    print("  %-30s cos=%.6f  relL2=%.5f" % (label, cos, rel))
    return cos, rel


def top1_share(A, B, label):
    D = (A - B).astype(np.float32)
    s = np.linalg.svd(D, compute_uv=False)
    e = s ** 2
    sh = float(e[0] / e.sum())
    print("  %-30s D: rows=%d  sigma1^2/sum=%.5f  s[:5]=%s"
          % (label, D.shape[0], sh, np.round(s[:5], 3)))
    return sh


if __name__ == "__main__":
    H, PAYLOAD = nf_header()
    globals()["PAYLOAD"] = PAYLOAD
    print("ninfer payload_offset =", PAYLOAD)
    idx = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]

    # ---------- 1) token embedding: HF BF16  vs  NF FP8-row-scaled ----------
    name = "model.language_model.embed_tokens.weight"
    shard = idx[name]
    e, off = hf_tensor_header(shard, name)
    print("=== token_embedding ===")
    print("  HF  :", name, e["dtype"], e["shape"], "shard", shard, "offset", off)
    need = off + ROWS * 5120 * 2
    if os.path.getsize(os.path.join(HF, shard)) < need:
        print("  SKIP: shard not fully downloaded (need %d, have %d)"
              % (need, os.path.getsize(os.path.join(HF, shard))))
    else:
        A = np.frombuffer(hf_read(os.path.join(HF, shard), off, ROWS * 5120 * 2),
                          dtype="<f2").reshape(ROWS, 5120).astype(np.float32)
        B = nf_dequant_fp8_row("text/token_embedding", (248320, 5120), ROWS)
        metrics(A, B, "embed_tokens[0:256] (BF16 vs FP8)")
        top1_share(A, B, "embed_tokens[0:256]")

    # ---------- 2) abliteration-sensitive: o_proj / down_proj (NVFP4 vs FP8) ----------
    print("\n=== abliteration-sensitive tensors ===")
    print("(HF side arrives once model-00002-of-00002.safetensors is downloaded)")
    pairs = [
        ("model.language_model.layers.3.self_attn.o_proj.weight",
         "text/layers/3/attention/output", (5120, 6144)),
        ("model.language_model.layers.3.mlp.down_proj.weight",
         "text/layers/3/mlp/down", (5120, 17408)),
        ("model.language_model.layers.0.mlp.down_proj.weight",
         "text/layers/0/mlp/down", (5120, 17408)),
    ]
    for hfname, nfname, shape in pairs:
        shard = idx.get(hfname)
        if shard is None:
            print("  %-52s <not in index>" % hfname)
            continue
        p = os.path.join(HF, shard)
        if not os.path.exists(p):
            print("  %-52s <shard missing: %s>" % (hfname, shard))
            continue
        e, off = hf_tensor_header(shard, hfname)
        n, k = shape
        nbytes = ROWS * (k // 2) + ROWS * (k // 16) + 4  # codes + scales + divisor
        if os.path.getsize(p) < off + ROWS * (k // 2):
            print("  %-52s <shard incomplete>" % hfname)
            continue
        codes = np.frombuffer(hf_read(p, off, ROWS * (k // 2)),
                              dtype=np.uint8).reshape(ROWS, k // 2)
        so = off + n * (k // 2) if "weight_scale" not in e else off
        # HF layout: [out, in//2] codes, then [out, in//16] scales, then scalar
        sc_off = off + n * (k // 2)
        scales = fp8_e4m3fn_to_f32(
            np.frombuffer(hf_read(p, sc_off, ROWS * (k // 16)),
                          dtype=np.uint8).reshape(ROWS, k // 16))
        div_off = sc_off + n * (k // 16)
        div = struct.unpack("<f", hf_read(p, div_off, 4))[0]
        lo = codes & 0x0F
        hi = (codes >> 4) & 0x0F
        w = np.empty((ROWS, k), dtype=np.float32)
        w[:, 0::2] = E2M1[lo]
        w[:, 1::2] = E2M1[hi]
        w *= np.repeat(scales, 16, axis=1)
        w *= div
        print("  HF %s dtype=%s shape=%s divisor=%.6g" % (hfname, e["dtype"], e["shape"], div))
        nfo = nf_obj(H, nfname)
        if nfo is None:
            print("    ninfer object not found:", nfname)
            continue
        if nfo["format"] == "FP8_E4M3FN_ROW_BF16S":
            B = nf_dequant_fp8_row(nfname, (n, k), ROWS)
        elif nfo["format"] == "NVFP4":
            # NVFP4 payload: codes [n,k/2], swizzled scales, fp32 divisor
            raw = nf_read(nfo["offset"], nfo["bytes"])
            csz = n * (k // 2)
            cs = np.frombuffer(raw[:csz], dtype=np.uint8).reshape(n, k // 2)[:ROWS]
            lo = cs & 0x0F
            hi = (cs >> 4) & 0x0F
            B = np.empty((ROWS, k), dtype=np.float32)
            B[:, 0::2] = E2M1[lo]
            B[:, 1::2] = E2M1[hi]
            print("    ninfer NVFP4 codes read; scales swizzled -> skipping magnitude scale")
            continue
        else:
            print("    ninfer format not handled:", nfo["format"])
            continue
        metrics(w, B, "%s%s" % (hfname.split("layers.")[1], "  (NVFP4 vs %s)" % nfo["format"]))
        top1_share(w, B, hfname.split("layers.")[1])
