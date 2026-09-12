#!/usr/bin/env python3
"""Does the dflash2 artifact ship a dedicated proposal/draft head we are ignoring?"""
import json
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as handle:
    handle.read(8)
    (length,) = struct.unpack("<Q", handle.read(8))
    doc = json.loads(handle.read(length).decode("utf-8"))

entries = doc["objects"]
print("===== text/ entries mentioning head / draft / proposal =====")
for item in entries:
    n = item["name"]
    low = n.lower()
    if n.startswith("text/") and ("head" in low or "draft" in low or "proposal" in low
                                 or "lm_head" in low):
        print(f"  {n:<58} {str(item.get('shape')):<22} {item.get('format')}")

print()
print("===== all entries mentioning 'head' anywhere =====")
for item in entries:
    n = item["name"]
    if "head" in n.lower():
        print(f"  {n:<58} {str(item.get('shape')):<22} {item.get('format')}")

print()
print("===== top-level text/ key histogram (first 2 segments) =====")
import collections
hist = collections.Counter()
for item in entries:
    n = item["name"]
    if n.startswith("text/"):
        parts = n.split("/")
        hist["/".join(parts[:2])] += 1
for k, v in sorted(hist.items()):
    print(f"  {k:<40} {v}")
