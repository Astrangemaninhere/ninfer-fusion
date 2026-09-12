#!/usr/bin/env python3
"""Classify every dumped layer plane as written / zeroed / partial."""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/kvdump_all"
rows = []
for meta_path in sorted(glob.glob(os.path.join(DUMP, "kvc_0_*_L*_meta.txt"))):
    pre = meta_path[: -len("_meta.txt")]
    layer = int(os.path.basename(pre).split("_L")[1])
    stats = {}
    for plane in ("k", "v", "ks", "vs"):
        p = pre + "_" + plane + ".bin"
        if not os.path.exists(p):
            stats[plane] = -1
            continue
        raw = np.fromfile(p, dtype=np.uint8)
        stats[plane] = int((raw != 0).sum())
    rows.append((layer, stats))
rows.sort()
print("layer  k_nz    v_nz    ks_nz  vs_nz")
for layer, s in rows:
    print("%5d  %6d  %6d  %5d  %5d%s" % (layer, s["k"], s["v"], s["ks"], s["vs"],
                                         "   <== k zero" if s["k"] == 0 else
                                         ("   <== v zero" if s["v"] == 0 else "")))
