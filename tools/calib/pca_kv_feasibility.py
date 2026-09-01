#!/usr/bin/env python3
# PCA + entropy feasibility for NVFP4 K/V code nibbles.
#
# Reproduces the production post-rotation code distribution, then tests
# per-16-channel PCA rotations and reports rANS stream sizes for the fixed
# slot codec (16 streams of 512 nibbles per half, shared per-half frequencies).
import argparse
import glob
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys_path = Path(__file__).resolve().parents[2] / "tools" / "calib"
sys.path.insert(0, str(sys_path))
import rans_nvfp4 as rans  # noqa: E402


def load_kv(path):
    raw = Path(path).read_bytes()
    hdr = struct.unpack("<16s6I2i4I", raw[:64])
    tokens = hdr[5]
    arr = np.frombuffer(raw, dtype="<u2", offset=64 + tokens * 4).view(np.float16).astype(np.float32)
    half = 256 * 4 * tokens
    k = arr[:half].reshape((256, 4, tokens), order="F").transpose(2, 1, 0)
    v = arr[half : 2 * half].reshape((256, 4, tokens), order="F").transpose(2, 1, 0)
    return k, v


def load_so4(path):
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    for block in re.findall(r"\{\{([^}]*)\},\s*\{([^}]*)\},\s*\{([^}]*)\},\s*\{([^}]*)\}\}", text):
        mat = []
        for row in block:
            mat.append([float(x) for x in re.findall(r"([-+0-9.e]+)f", row)])
        rows.append(mat)
    return np.asarray(rows, dtype=np.float32)


def quantize_codes(x):
    x = np.nan_to_num(x.astype(np.float64))
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 6.0, 2.0**-9)
    q = np.abs(groups) / scale
    edges = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    codes = np.searchsorted(edges, q)
    codes = np.where(groups < 0, codes | 8, codes)
    return codes.reshape(*x.shape[:-1], -1).astype(np.uint8), scale


def slot_sizes(codes):
    """codes: [tokens, heads, 256]; returns max per-page stream size per half."""
    tokens, heads = codes.shape[0], codes.shape[1]
    pages = tokens // 64
    maxima = [[], []]
    for page in range(pages):
        for half in range(2):
            page_max = 0
            for head in range(heads):
                streams = []
                for stream in range(16):
                    block = np.concatenate(
                        [codes[page * 64 + half * 32 + stream * 2, head, :],
                         codes[page * 64 + half * 32 + stream * 2 + 1, head, :]])
                    streams.append(block)
                freqs = rans.build_freqs(np.concatenate(streams))
                sizes = [len(rans.encode(block, freqs)[0]) for block in streams]
                page_max = max(page_max, max(sizes))
            maxima[half].append(page_max)
    return maxima


def pca_basis(x, block):
    x = x.reshape(-1, x.shape[-1] // block, block).transpose(0, 1, 2).reshape(-1, block)
    cov = np.cov(x, rowvar=False)
    w, v = np.linalg.eigh(cov)
    return v[:, ::-1], w[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--limit-frames", type=int, default=4)
    ap.add_argument("--rot-table", default=str(Path(__file__).resolve().parents[2] /
                                               "ninfer" / "src" / "ops" / "kernel" /
                                               "gqa_isoquant_rot.cu"))
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.records) / "*.kvc")))[: args.limit_frames]

    so4 = load_so4(args.rot_table)
    frames = [load_kv(p) for p in files]
    k_all = np.concatenate([k for k, _ in frames], axis=0)
    v_all = np.concatenate([v for _, v in frames], axis=0)

    def apply_so4(x):
        out = np.empty_like(x)
        for block in range(64):
            base = block * 4
            out[:, :, base : base + 4] = np.einsum(
                "jk,thk->thj", so4[block], x[:, :, base : base + 4])
        return out

    def apply_basis(x, basis, block):
        out = np.empty_like(x)
        for b in range(x.shape[-1] // block):
            base = b * block
            out[:, :, base : base + block] = np.einsum(
                "jk,thk->thj", basis, x[:, :, base : base + block])
        return out

    for name, x in [("K", k_all), ("V", v_all)]:
        so4_codes, _ = quantize_codes(apply_so4(x))
        so4_sizes = slot_sizes(so4_codes)
        print(f"{name} SO(4): max={max(so4_sizes[0] + so4_sizes[1])} "
              f"h0={max(so4_sizes[0])} h1={max(so4_sizes[1])} "
              f"over166={max(so4_sizes[0] + so4_sizes[1]) > 166}")

        for block in (16, 64):
            basis, _ = pca_basis(apply_so4(x), block)
            pca_codes, _ = quantize_codes(apply_basis(apply_so4(x), basis, block))
            pca_sizes = slot_sizes(pca_codes)
            print(f"{name} SO(4)+PCA{block}: max={max(pca_sizes[0] + pca_sizes[1])} "
                  f"h0={max(pca_sizes[0])} h1={max(pca_sizes[1])} "
                  f"over166={max(pca_sizes[0] + pca_sizes[1]) > 166}")

            raw_basis, _ = pca_basis(x, block)
            raw_codes, _ = quantize_codes(apply_basis(x, raw_basis, block))
            raw_sizes = slot_sizes(raw_codes)
            print(f"{name} PCA{block}: max={max(raw_sizes[0] + raw_sizes[1])} "
                  f"h0={max(raw_sizes[0])} h1={max(raw_sizes[1])} "
                  f"over166={max(raw_sizes[0] + raw_sizes[1]) > 166}")


if __name__ == "__main__":
    main()
