#!/usr/bin/env python3
"""Blame partition for the dflash2 collapse.

Speculative decoding guarantees the licensed token stream == the target's own greedy
stream, so round offsets are exactly the cumulative licensed counts. With that anchor
we can ask, per column: was the TARGET's argmax wrong, or was the DRAFT wrong?

  correct[j] = (argmax[j] == ground_truth[t + j])   -> target's verify block column j
  draftok[j] = (draft[j]  == ground_truth[t + j])   -> draft block column j
"""
import pathlib
import re
import sys

ROUND = re.compile(
    r"\[df2dbg\] row=(?P<row>-?\d+) anchor=(?P<anchor>-?\d+) frontier=(?P<frontier>-?\d+) "
    r"valid=(?P<valid>-?\d+) in_extent=(?P<in_extent>-?\d+) out_extent=(?P<out_extent>-?\d+) "
    r"accepted=(?P<accepted>-?\d+) count=(?P<count>-?\d+)")
COL = re.compile(
    r"\[df2dbg\]\s+col=(?P<col>-?\d+) pos=(?P<pos>-?\d+) verify=(?P<verify>-?\d+) "
    r"draft=(?P<draft>-?\d+) argmax=(?P<argmax>-?\d+)")


def parse(path):
    rounds, cur = [], None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            cur = {k: int(m.group(k)) for k in
                   ("row", "anchor", "frontier", "valid", "in_extent", "out_extent",
                    "accepted", "count")}
            cur["cols"] = []
            rounds.append(cur)
            continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({k: int(m.group(k))
                                for k in ("col", "pos", "verify", "draft", "argmax")})
    return rounds


def plain_ids(path):
    ids = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            ids = [int(x) for x in line.split("generated ids", 1)[1].split()]
    return ids


rounds = parse("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_probe_raw.log")
gt = plain_ids("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_plain.log")
print(f"rounds={len(rounds)}  gt_tokens={len(gt)}")

# --- exact offsets from the cumulative licensed counts -----------------------
offset = 0
for r in rounds:
    r["t"] = offset
    offset += max(r["count"], 0)
print(f"total licensed tokens implied by the probe: {offset}")
print()

# self-check: the probe's own column-0 argmax must equal the stream at offset t
chk = sum(1 for r in rounds if r["cols"] and r["cols"][0]["argmax"] ==
          (gt[r["t"]] if r["t"] < len(gt) else -1))
print(f"self-check argmax[0] == gt[t] : {chk}/{len(rounds)} rounds")
print()

drafting = [r for r in rounds if r["in_extent"] > 0]
print(f"drafting rounds: {len(drafting)}")
print()

W = max(len(r["cols"]) for r in rounds)
correct = [0] * W          # target argmax == ground truth
draft_ok = [0] * W         # draft == ground truth
agreed = [0] * W           # argmax == draft (what the engine counts)
n = [0] * W                # samples
for r in drafting:
    for c in r["cols"]:
        j = c["col"]
        t = r["t"] + j
        if j >= W or t >= len(gt):
            continue
        n[j] += 1
        if c["argmax"] == gt[t]:
            correct[j] += 1
        if c["draft"] != -1 and c["draft"] == gt[t]:
            draft_ok[j] += 1
        if c["draft"] != -1 and c["argmax"] == c["draft"]:
            agreed[j] += 1

print("per column:  n | target argmax correct | draft correct | argmax==draft")
print(f"  {'col':>3} {'n':>4} {'target_ok':>10} {'draft_ok':>9} {'agreed':>7} "
      f"{'target%':>8} {'draft%':>7}")
for j in range(W):
    if not n[j]:
        continue
    print(f"  {j:>3} {n[j]:>4} {correct[j]:>10} {draft_ok[j]:>9} {agreed[j]:>7} "
          f"{100.0*correct[j]/n[j]:>7.1f}% {100.0*draft_ok[j]/n[j]:>6.1f}%")

print()
print("=== chain conditionals (only rounds with in_extent>0) ===")
kept0 = [r for r in drafting if r["cols"] and r["t"] + 1 < len(gt)
         and r["cols"][0]["argmax"] == gt[r["t"]]]
print(f"rounds where the TARGET got column 0 right: {len(kept0)}")
if kept0:
    tgt1 = sum(1 for r in kept0
               if len(r["cols"]) > 1 and r["t"] + 1 < len(gt)
               and r["cols"][1]["argmax"] == gt[r["t"] + 1])
    drf1 = sum(1 for r in kept0
               if len(r["cols"]) > 1 and r["cols"][1]["draft"] != -1
               and r["t"] + 1 < len(gt) and r["cols"][1]["draft"] == gt[r["t"] + 1])
    print(f"  of those, TARGET column 1 also right : {tgt1}   "
          f"(healthy reference ~82%)")
    print(f"  of those, DRAFT  column 1 right      : {drf1}")

acc0 = [r for r in drafting if r["accepted"] >= 1]
print()
print(f"rounds where the engine accepted column 0: {len(acc0)}")
if acc0:
    t1 = sum(1 for r in acc0 if len(r["cols"]) > 1 and r["t"] + 1 < len(gt)
             and r["cols"][1]["argmax"] == gt[r["t"] + 1])
    d1 = sum(1 for r in acc0 if len(r["cols"]) > 1 and r["cols"][1]["draft"] != -1
             and r["t"] + 1 < len(gt) and r["cols"][1]["draft"] == gt[r["t"] + 1])
    print(f"  TARGET column 1 right: {t1}/{len(acc0)}")
    print(f"  DRAFT  column 1 right: {d1}/{len(acc0)}")

print()
print("=== the first 8 drafting rounds, aligned to ground truth ===")
for r in drafting[:8]:
    t = r["t"]
    print(f"  t={t} anchor={r['anchor']} in={r['in_extent']} accepted={r['accepted']} "
          f"count={r['count']}")
    print(f"    gt      : {gt[t:t+W]}")
    print(f"    col     : {[c['col'] for c in r['cols']]}")
    print(f"    verify  : {[c['verify'] for c in r['cols']]}")
    print(f"    draft   : {[c['draft'] for c in r['cols']]}")
    print(f"    argmax  : {[c['argmax'] for c in r['cols']]}")
    marks = ["T" if (t + c["col"] < len(gt) and c["argmax"] == gt[t + c["col"]]) else "."
             for c in r["cols"]]
    print(f"    tgt_ok  : {''.join(marks)}")
