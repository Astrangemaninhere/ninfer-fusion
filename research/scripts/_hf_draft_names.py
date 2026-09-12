#!/usr/bin/env python3
"""Locate the draft tensors in the HF checkpoint (names, dtypes, shapes)."""
import json
import pathlib
import struct

HF_DIR = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")
print("files in HF dir:")
for p in sorted(HF_DIR.iterdir()):
    print("   ", p.name, p.stat().st_size if p.is_file() else "(dir)")

names = {}
for path in sorted(HF_DIR.glob("*.safetensors")):
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
    for name, meta in header.items():
        if name != "__metadata__":
            names[name] = (meta["dtype"], meta["shape"])
print(f"\ntotal tensors: {len(names)}")
print("\nfirst 20 names:")
for n in list(names)[:20]:
    print("   ", n, names[n])

for pat in ("draft", "selector", "codebook", "markov", "conv", "context_key", "feature"):
    hits = [(n, v) for n, v in names.items() if pat in n.lower()]
    print(f"\n=== names containing '{pat}': {len(hits)} ===")
    for n, v in hits[:14]:
        print("   ", n, v)
