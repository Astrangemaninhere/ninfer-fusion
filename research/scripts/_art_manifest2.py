#!/usr/bin/env python3
"""Second pass: the draft/speculative subtree of the dflash2 artifact, plus identity."""
import json
import struct
import sys

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"


def load(path):
    with open(path, "rb") as handle:
        handle.read(8)
        (length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(length)
    return json.loads(raw.decode("utf-8"))


doc = load(ART)
print("===== identity =====")
print(json.dumps(doc.get("identity"), indent=2, ensure_ascii=False)[:4000])

objs = doc["objects"]
print()
print("===== top-level groups under objects =====")
groups = {}
for key in objs:
    head = key.split("/")[0]
    groups[head] = groups.get(head, 0) + 1
for head, count in sorted(groups.items()):
    print(f"  {head:<28} {count} tensors")

print()
for head in groups:
    if head in ("text",):
        continue
    print(f"===== objects['{head}'] : all entries =====")
    for key in sorted(objs):
        if key.startswith(head + "/"):
            entry = objs[key]
            if isinstance(entry, dict):
                shape = entry.get("shape")
                dtype = entry.get("dtype")
                fmt = entry.get("format") or entry.get("layout")
                extra = {k: v for k, v in entry.items()
                         if k not in ("shape", "dtype", "format", "layout", "offset", "bytes")}
                print(f"  {key:<62} {str(shape):<22} {dtype} {fmt} {extra if extra else ''}")
            else:
                print(f"  {key:<62} {entry}")
    print()
