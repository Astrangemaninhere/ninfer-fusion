#!/usr/bin/env python3
"""Check a KV dump for plane aliasing between layers (address ranges) and
byte-identical planes (content), plus a NaN/0xFF survey of each plane."""
import glob
import os
import re
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_e8"


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "addr":
                # "addr k=<hex> v=<hex> ks=<hex> vs=<hex> bytes k=<n> ..."
                fields = {}
                section = "addr"
                for tok in parts[1:]:
                    if tok == "bytes":
                        section = "bytes"
                        continue
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        fields[section + ":" + k] = v
                meta["addr"] = fields
            else:
                meta[parts[0]] = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    return meta


def main():
    metas = sorted(glob.glob(os.path.join(DUMP, "kvc_*_L*_meta.txt")))
    if not metas:
        print("no layer metas in", DUMP)
        return
    spans = []   # (layer, plane, start, end)
    seen = set()
    for p in metas:
        meta = read_meta(p)
        head = None
        for key in meta:
            if key.startswith("layer="):
                head = meta[key]
                head["layer"] = key.split("=", 1)[1]
                break
        if head is None:
            head = meta.get("layer", {})
        layer = int(head["layer"])
        if "addr" not in meta:
            print("L%d: no addr line (probe predates the pointer print)" % layer)
            continue
        addr = meta["addr"]
        for plane in ("k", "v", "ks", "vs"):
            a = addr.get("addr:" + plane)
            n = addr.get("bytes:" + plane)
            if a is None or n is None or a == "(nil)":
                continue
            key = (layer, plane)
            if key in seen:
                continue
            seen.add(key)
            start = int(a, 16)
            spans.append((layer, plane, start, start + int(n)))
    print("planes with addresses: %d" % len(spans))
    overlaps = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            l1, p1, s1, e1 = spans[i]
            l2, p2, s2, e2 = spans[j]
            if s1 < e2 and s2 < e1:
                overlaps += 1
                if overlaps <= 12:
                    print("  OVERLAP L%d.%s [0x%x,0x%x) vs L%d.%s [0x%x,0x%x) = %d bytes"
                          % (l1, p1, s1, e1, l2, p2, s2, e2,
                             min(e1, e2) - max(s1, s2)))
    print("total overlapping plane pairs: %d" % overlaps)
    # content survey + identical-plane detection
    data = {}
    for p in sorted(glob.glob(os.path.join(DUMP, "kvc_*_L*_k.bin"))):
        base = os.path.basename(p)
        layer = int(base.split("_L")[1].split("_")[0])
        raw = np.fromfile(p, dtype=np.uint8)
        data[layer] = raw
        print("L%d k.bin bytes=%d ff=%d zero=%d" % (layer, raw.size,
                                                    int((raw == 0xFF).sum()), int((raw == 0).sum())))
    layers = sorted(data)
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            a, b = data[layers[i]], data[layers[j]]
            n = min(a.size, b.size)
            if n and np.array_equal(a[:n], b[:n]):
                print("  IDENTICAL content L%d vs L%d (%d bytes) -> aliased or same data"
                      % (layers[i], layers[j], n))


if __name__ == "__main__":
    main()
