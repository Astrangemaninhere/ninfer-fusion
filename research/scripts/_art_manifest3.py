#!/usr/bin/env python3
"""Third pass: walk the nested manifest and print the draft subtree with shapes."""
import json
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"


def load(path):
    with open(path, "rb") as handle:
        handle.read(8)
        (length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(length)
    return json.loads(raw.decode("utf-8"))


doc = load(ART)
objs = doc["objects"]
print("top-level groups:", sorted(objs.keys()))
print()
for group in sorted(objs.keys()):
    if group == "text":
        continue
    sub = objs[group]
    print(f"===== objects[{group!r}] =====")
    if isinstance(sub, dict) and not any(k in sub for k in ("shape", "dtype")):
        for name in sorted(sub.keys()):
            entry = sub[name]
            if isinstance(entry, dict):
                shape = entry.get("shape")
                dtype = entry.get("dtype")
                fmt = entry.get("format") or entry.get("layout")
                extra = {k: v for k, v in entry.items()
                         if k not in ("shape", "dtype", "format", "layout", "offset", "bytes")}
                if shape is None and extra:
                    # one more level of nesting
                    print(f"  {name}:")
                    for n2 in sorted(entry.keys()):
                        e2 = entry[n2]
                        if isinstance(e2, dict):
                            print(f"    {n2:<58} {str(e2.get('shape')):<20} {e2.get('dtype')} "
                                  f"{e2.get('format') or e2.get('layout') or ''}")
                        else:
                            print(f"    {n2:<58} {e2}")
                else:
                    print(f"  {name:<60} {str(shape):<20} {dtype} {fmt or ''} "
                          f"{extra if extra else ''}")
            else:
                print(f"  {name:<60} {entry}")
    else:
        print(json.dumps(sub, indent=2)[:3000])
    print()

print("===== text-side tensor count =====")
text = objs.get("text", {})
print("text keys sample:", sorted(text.keys())[:12] if isinstance(text, dict) else type(text))
