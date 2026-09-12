#!/usr/bin/env python3
"""Dataset composition + per-source teacher quality (uses only tokens/ids16)."""
import collections
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/bench/hs_cache_topk2"
CORPUS = "/home/user/bench/hs_corpus.jsonl"

recs = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
stat = collections.defaultdict(lambda: {"n": 0, "tokens": 0, "top1": 0, "top4": 0, "top16": 0, "pos": 0})
for p in sorted(glob.glob(os.path.join(OUT, "seq_*.npz"))):
    seq = int(os.path.basename(p)[4:10])
    if seq >= len(recs):
        continue
    src = recs[seq]["src"].split(":")[0]
    with np.load(p) as z:
        tok = z["tokens"]
        ids16 = z["ids16"]
        if tok.size < 2:
            continue
        nxt = tok[1:]
        hit = ids16[:-1] == nxt[:, None]
        s = stat[src]
        s["n"] += 1
        s["tokens"] += tok.size
        s["top1"] += int(hit[:, 0].sum())
        s["top4"] += int(hit[:, :4].any(axis=1).sum())
        s["top16"] += int(hit.any(axis=1).sum())
        s["pos"] += nxt.size
tot = sum(s["tokens"] for s in stat.values())
print("source   files  tokens   share  top1   top4   top16")
for src, s in sorted(stat.items(), key=lambda kv: -kv[1]["tokens"]):
    print("%-8s %5d %7d %6.1f%%  %.3f  %.3f  %.3f"
          % (src, s["n"], s["tokens"], 100.0 * s["tokens"] / tot,
             s["top1"] / s["pos"], s["top4"] / s["pos"], s["top16"] / s["pos"]))
print("TOTAL    %5d %7d" % (sum(s["n"] for s in stat.values()), tot))
