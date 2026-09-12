#!/usr/bin/env python3
"""Print ONLY the dflash2/ tensor group (the draft checkpoint's real inventory)."""
import json
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"

with open(ART, "rb") as handle:
    handle.read(8)
    (length,) = struct.unpack("<Q", handle.read(8))
    doc = json.loads(handle.read(length).decode("utf-8"))

entries = [e for e in doc["objects"] if str(e.get("name", "")).startswith("dflash2/")]
print(f"dflash2 tensors: {len(entries)}")
print()
for item in entries:
    name = item["name"]
    print(f"  {name:<58} {str(item.get('shape')):<24} {item.get('format')} "
          f"{item.get('layout')}")

print()
print("===== non-dflash2 top-level groups for reference =====")
names = [e["name"] for e in doc["objects"]]
import collections
print(collections.Counter(n.split("/")[0] for n in names))
print()
print("===== mtp/ + any dflash/ group =====")
for item in doc["objects"]:
    n = item["name"]
    if n.startswith("dflash/") or n.startswith("dflash2/") or n.startswith("spec"):
        print(f"  {n:<58} {str(item.get('shape')):<24} {item.get('format')}")
