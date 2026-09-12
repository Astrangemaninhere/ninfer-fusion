#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z1: print the vLLM-vs-ninfer side-by-side acceptance table.

Pure stdlib; runs under WSL python3 or Windows python.

Metric alignment (no fudge factors needed for the headline):
  vLLM  acceptance_rate = accepted_tokens / draft_tokens
  ninfer acceptance     = accepted_tokens / drafted_tokens
  -> same denominator; vLLM's `num_drafts` is ninfer's `rounds`.

Per-position (vLLM prints RATES, ninfer prints COUNTS):
  p_i                  = accepted_by_pos[i] / rounds              (ninfer -> rate)
  accepted_by_pos[i]   = p_i * num_drafts                         (vLLM -> counts)
"""
import json
import os
import sys

RESULT = "/mnt/c/Users/User/Documents/ziqinzhang/_z1_result.json"
if os.name == "nt":
    RESULT = r"C:\Users\User\Documents\ziqinzhang\_z1_result.json"

# ninfer side, verbatim from _collab/build/T1_dspark_pos0_only.md
NINFER = [
    {"tag": "ninfer dspark K=7 OLD (pre-A5b)", "log": "/home/user/pfv_dspark_OLD.log",
     "rounds": 73, "drafted": 171, "accepted": 21,
     "by_pos": [20, 1, 0, 0, 0, 0, 0], "note": "36-token prompt, greedy, --draft-tokens 7"},
    {"tag": "ninfer dspark K=7 NEW (post-A5b)", "log": "/home/user/pfv_dspark_NEW.log",
     "rounds": 76, "drafted": 236, "accepted": 18,
     "by_pos": [17, 1, 0, 0, 0, 0, 0], "note": "同 prompt；base_dspark7_r1/r2/r3 三次逐位相同"},
    {"tag": "ninfer dspark K=7 (08-26 build, temp=1)", "log": "/home/user/nr-ds-nosvip.log",
     "rounds": 87, "drafted": 603, "accepted": 82,
     "by_pos": [40, 23, 15, 4, 0, 0, 0], "note": "4964-token prompt / temp 1.0 — 非贪心，仅示健康剖面"},
    {"tag": "ninfer dspark K=7 (08-26 build, qual)", "log": "/home/user/nr-ds-qual.log",
     "rounds": 1395, "drafted": 9715, "accepted": 2699,
     "by_pos": [937, 637, 438, 302, 196, 125, 64], "note": "同 08-26 栈；非贪心"},
]


def pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def main():
    if not os.path.exists(RESULT):
        print("NO RESULT FILE:", RESULT)
        print("run _z1_run.sh (or _z1_measure.py) first.")
        return 2
    r = json.load(open(RESULT, encoding="utf-8"))
    d = r["delta"]
    nd, ndt, na = d["num_drafts"], d["num_draft_tokens"], d["num_accepted_tokens"]
    npos = len(d["per_pos_counts"])

    print("=" * 100)
    print("vLLM DSpark acceptance measurement")
    print("=" * 100)
    print("cmd            :", " ".join(r["cmd"]))
    print("prompt_tokens  :", r["prompt_tokens"], "  completion_tokens:", r["completion_tokens"])
    print("wall_time_s    : %.2f" % r["wall_time_s"])
    print()
    print("-- raw counters (delta between /metrics scrapes) --")
    print("  num_drafts        =", nd)
    print("  num_draft_tokens  =", ndt)
    print("  num_accepted      =", na)
    print("  acceptance_rate   = %.4f %%   (accepted/draft_tokens -- vLLM headline)" % pct(na, ndt))
    print("  acceptance_length = %.4f      (1 + accepted/drafts)" % (1.0 + (na / nd) if nd else 0))
    print("  per_pos counts    =", d["per_pos_counts"])
    print()

    print("=" * 100)
    print("SIDE BY SIDE  (same denominators: accepted/drafted ==  accepted/draft_tokens)")
    print("=" * 100)
    hdr = ("%-42s %7s %8s %8s %8s %9s   %s"
           % ("run", "rounds", "drafted", "accepted", "acc%", "acc_len", "by-pos counts"))
    print(hdr)
    print("-" * len(hdr))

    print("%-42s %7d %8d %8d %7.2f%% %9.4f   %s"
          % ("vLLM 0.26.0 dspark K=7 (THIS RUN)",
             nd, ndt, na, pct(na, ndt),
             1.0 + (na / nd if nd else 0),
             ",".join(str(x) for x in d["per_pos_counts"])))
    for n in NINFER:
        print("%-42s %7d %8d %8d %7.2f%% %9.4f   %s"
              % (n["tag"], n["rounds"], n["drafted"], n["accepted"],
                 pct(n["accepted"], n["drafted"]),
                 1.0 + (n["accepted"] / n["rounds"] if n["rounds"] else 0),
                 ",".join(str(x) for x in n["by_pos"])))
    print()

    print("=" * 100)
    print("PER-POSITION PROFILE, normalised to RATE p_i = accepted_at_pos_i / rounds")
    print("  (ninfer prints counts -> p_i = by_pos[i]/rounds ;  vLLM prints the rate directly)")
    print("=" * 100)
    widths = max(7, npos)
    head = "%-42s " % "run" + " ".join("  p%-5d" % i for i in range(widths))
    print(head)
    print("-" * len(head))
    print("%-42s " % "vLLM 0.26.0 dspark K=7 (THIS RUN)"
          + " ".join("%7.2f" % x for x in d["per_pos_rate_pct"]))
    for n in NINFER:
        rates = [pct(n["by_pos"][i] if i < len(n["by_pos"]) else 0, n["rounds"])
                 for i in range(widths)]
        print("%-42s " % n["tag"] + " ".join("%7.2f" % x for x in rates))
    print()

    print("=" * 100)
    print("VERDICT on 'positions >= 1 are structurally zero'")
    print("=" * 100)
    vpos = d["per_pos_counts"]
    nz_v = [i for i in range(1, len(vpos)) if vpos[i] > 0]
    print("vLLM p1..p%d counts = %s  -> nonzero at %s"
          % (len(vpos) - 1, vpos[1:], nz_v if nz_v else "NONE"))
    print("ninfer NEW counts   = [17, 1, 0, 0, 0, 0, 0] -> nonzero at [1] only")
    print("ninfer OLD counts   = [20, 1, 0, 0, 0, 0, 0] -> nonzero at [1] only")
    if nz_v:
        print("=> vLLM DOES accept at positions >=1 under this prompt: the ninfer")
        print("   per-position collapse is NOT an intrinsic property of DSpark.")
    else:
        print("=> vLLM also shows ~zero at positions >=1 for this prompt; compare with the")
        print("   08-26-stack ninfer runs (40,23,15,4 / 937,637,438,302,196,125,64) which")
        print("   DID accept at deep positions, i.e. the collapse is build/prompt dependent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
