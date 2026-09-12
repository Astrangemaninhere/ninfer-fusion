#!/usr/bin/env python3
"""Coarse map of which bytes of each dumped plane are non-zero, per layer."""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump"
for meta_path in sorted(glob.glob(os.path.join(DUMP, "kvc_0_*_L*_meta.txt"))):
    pre = meta_path[: -len("_meta.txt")]
    print("=== %s" % os.path.basename(pre))
    for plane in ("k", "v", "ks", "vs"):
        p = pre + "_" + plane + ".bin"
        if not os.path.exists(p):
            continue
        raw = np.fromfile(p, dtype=np.uint8)
        nz = raw != 0
        idx = np.flatnonzero(nz)
        first = int(idx[0]) if idx.size else -1
        last = int(idx[-1]) if idx.size else -1
        # coarse 16-bucket occupancy
        buckets = np.array_split(nz, 16)
        occ = "".join("%X" % min(15, int(b.sum() / max(1, b.size) * 15)) for b in buckets)
        print("  %-2s bytes=%7d nonzero=%7d first=%7d last=%7d occ=%s"
              % (plane, raw.size, nz.sum(), first, last, occ))
