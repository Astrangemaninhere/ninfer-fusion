#!/usr/bin/env python3
"""Compare the collected top-1 next-token agreement against the pilot pack."""
import glob
import os

import numpy as np

for pat, label in ((r"/mnt/c/Users/User/Documents/ziqinzhang/data/df2pilot/packs/seq_*.npz", "pilot"),
                   ("/home/user/bench/hs_cache_topk2/seq_*.npz", "new")):
    files = sorted(glob.glob(pat))
    if not files:
        print(label, "-> none")
        continue
    hits = []
    for p in files[:10]:
        with np.load(p) as z:
            if "top1" not in z.files:
                continue
            t = z["tokens"]
            o = z["top1"]
            if t.size > 1:
                hits.append(float((o[:-1] == t[1:]).mean()))
    print("%s: files=%d mean top1-next-token hit=%.3f (per file %s)"
          % (label, len(files), float(np.mean(hits)) if hits else -1,
             [round(h, 2) for h in hits[:6]]))
