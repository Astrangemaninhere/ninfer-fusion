#!/usr/bin/env python3
"""Pass-by-pass NaN onset for the Muse headdbg dumps (bf16 vs nvfp4).

A "pass" is one headdbg sweep (starts at L00_attn). The interesting question is
not "which layer is NaN" but "at which pass does it start, and at which layer
within that pass" — because a single poisoned step poisons everything after.
"""
import re

for tag in ("bf16", "nvfp4"):
    path = "/tmp/muse_kv_%s.err" % tag
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError as exc:
        print("%s: no file (%s)" % (tag, exc))
        continue

    passes, cur = [], None
    for ln in lines:
        m = re.search(r"\[headdbg\] (L\d+)_(\w+)\s+(.*)", ln)
        if not m:
            continue
        layer, sub, rest = m.group(1), m.group(2), m.group(3)
        if layer == "L00" and sub == "attn":
            cur = {"first_nan": None, "n": 0, "head": rest[:46]}
            passes.append(cur)
        if cur is None:
            continue
        cur["n"] += 1
        if cur["first_nan"] is None and "nan" in rest.lower():
            cur["first_nan"] = "%s_%s" % (layer, sub)

    print("=== %s: %d passes, %d headdbg lines ===" % (tag, len(passes), sum(p["n"] for p in passes)))
    for i, p in enumerate(passes):
        print("  pass %-2d lines=%-4d first_nan=%-12s L00_attn=%s"
              % (i, p["n"], p["first_nan"] or "none", p["head"]))
