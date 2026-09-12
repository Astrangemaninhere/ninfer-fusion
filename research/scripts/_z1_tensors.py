#!/usr/bin/env python3
"""Z1: enumerate safetensors tensor names/dtypes without loading torch."""
import json, struct, sys, collections, os

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h

def report(path, max_show=200):
    h = header(path)
    print("### %s" % path)
    print("  tensors: %d" % len(h))
    dt = collections.Counter()
    tot = 0
    for k, v in h.items():
        dt[v["dtype"]] += 1
        nb = 1
        for d in v["shape"]:
            nb *= d
        tot += nb * {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I8": 1,
                     "U8": 1, "I32": 4, "I64": 8}.get(v["dtype"], 2)
    print("  dtypes: %s" % dict(dt))
    print("  approx bytes: %.3f GiB" % (tot / 2**30))
    names = sorted(h)
    for k in names[:max_show]:
        print("    %-72s %-8s %s" % (k, h[k]["dtype"], h[k]["shape"]))
    if len(names) > max_show:
        print("    ... (%d more)" % (len(names) - max_show))
    print()
    return h

if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            report(p)
        except Exception as e:
            print("### %s\n  ERROR %s\n" % (p, e))
