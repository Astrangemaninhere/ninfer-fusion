#!/usr/bin/env python
# CPU reference order-0 static rANS for NVFP4 E2M1 code nibbles.
# Feasibility gate for the entropy-coded cold KV pool.
import argparse
import glob
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NINFERKVCAL1\x00\x00\x00\x00"
SCALE_BITS = 12
SCALE = 1 << SCALE_BITS
MASK = SCALE - 1
BYTE_L = 1 << 23


def build_freqs(symbols):
    counts = np.bincount(symbols, minlength=16).astype(np.int64)
    freqs = np.maximum(1, np.round(counts / counts.sum() * SCALE).astype(np.int64))
    diff = int(SCALE - freqs.sum())
    while diff > 0:
        idx = int(np.argmax(counts))
        freqs[idx] += 1
        counts[idx] = max(0, counts[idx] - 1)
        diff -= 1
    while diff < 0:
        idx = int(np.argmax(np.where(freqs > 1, freqs, 0)))
        freqs[idx] -= 1
        diff += 1
    return freqs


def encode(symbols, freqs):
    start = np.concatenate(([0], np.cumsum(freqs)[:-1])).astype(np.int64)
    out = bytearray()
    x = BYTE_L
    for s in reversed(symbols):
        f = int(freqs[s])
        x_max = ((BYTE_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + int(start[s])
    out.extend(x.to_bytes(4, "little"))
    return bytes(out), start


def decode(data, freqs, start, count):
    x = int.from_bytes(data[-4:], "little")
    inb = bytearray(data[:-4])
    out = []
    for _ in range(count):
        slot = x & MASK
        s = int(np.searchsorted(start, slot, side="right") - 1)
        out.append(s)
        x = int(freqs[s]) * (x >> SCALE_BITS) + slot - int(start[s])
        while x < BYTE_L:
            x = (x << 8) | inb.pop()
    return out


def load_k(path):
    raw = Path(path).read_bytes()
    hdr = struct.unpack("<16s6I2i4I", raw[:64])
    if hdr[0] != MAGIC:
        raise ValueError(path)
    tokens = hdr[5]
    arr = np.frombuffer(raw, dtype="<u2", offset=64 + tokens * 4).view(np.float16).astype(np.float32)
    half = 256 * 4 * tokens
    k = arr[:half].reshape((256, 4, tokens), order="F").transpose(2, 1, 0)
    return k


def e2m1_nibbles(x):
    x = np.nan_to_num(x.astype(np.float64))
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 6.0, 2.0**-9)
    q = np.abs(groups) / scale
    edges = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    codes = np.searchsorted(edges, q)
    codes = np.where(groups < 0, codes | 8, codes)
    return codes.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--limit-frames", type=int, default=4)
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.records) / "*.kvc")))[: args.limit_frames]
    rows = []
    for path in files:
        k = load_k(path)
        for head in range(4):
            codes = e2m1_nibbles(k[:, head, :]).ravel()
            freqs = build_freqs(codes)
            raw, start = encode(codes, freqs)
            dec = decode(raw, freqs, start, len(codes))
            if not np.array_equal(dec, codes):
                raise SystemExit(f"roundtrip failed {path}")
            total_bits = len(raw) * 8 + 16 * 32  # stream + 16-entry frequency table
            page_ratio = (128.0 + 16.0) * 8 / (total_bits / k.shape[0] + 16 * 8)
            rows.append((Path(path).name, head, total_bits / len(codes), page_ratio))
    print("frame head rans-bits/nibble rans-page-ratio")
    for name, head, bits, ratio in rows:
        print(f"{name} {head} {bits:.3f} {ratio:.2f}x")


if __name__ == "__main__":
    main()
