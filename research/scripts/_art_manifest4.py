#!/usr/bin/env python3
"""Fourth pass: objects is a LIST of named entries. Print the draft subtree."""
import json
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"

with open(ART, "rb") as handle:
    handle.read(8)
    (length,) = struct.unpack("<Q", handle.read(8))
    doc = json.loads(handle.read(length).decode("utf-8"))

entries = doc["objects"]
print("entry count:", len(entries))
print("fields on entry[0]:", sorted(entries[0].keys()))
print("entry[0]:", json.dumps(entries[0], ensure_ascii=False)[:600])
print()

groups = {}
for item in entries:
    name = item.get("name", "?")
    head = name.split("/")[0]
    groups[head] = groups.get(head, 0) + 1
print("groups:", groups)
print()

for item in entries:
    name = item.get("name", "")
    if name.startswith("text/"):
        continue
    keep = {k: v for k, v in item.items() if k not in ("name",)}
    print(f"  {name:<64} {json.dumps(keep, ensure_ascii=False)[:300]}")
