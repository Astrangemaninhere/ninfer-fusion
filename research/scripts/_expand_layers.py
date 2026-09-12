#!/usr/bin/env python3
"""Expand a layer list like "10-15" or "3,7,12-14" to a comma list (for --kv-residual-layers)."""
import sys

out = []
for part in sys.argv[1].split(","):
    if "-" in part:
        lo, hi = part.split("-")
        out += [str(i) for i in range(int(lo), int(hi) + 1)]
    elif part:
        out.append(part)
print(",".join(out))
