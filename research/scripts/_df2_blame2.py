#!/usr/bin/env python3
"""Correctly aligned blame partition for the dflash2 collapse.

Facts established from the raw probe:
  * the block is [anchor, d0..d_{K-1}] at positions [s, s+1, ...], i.e. verify[0] ==
    the anchor and verify[j] == draft[j-1];
  * argmax[j] is the target's prediction for the token at stream index s + j + 1;
  * the engine's own `accepted`/`count` fields lag one round.

Ground truth: speculative decoding only ever licenses a token equal to the target's
argmax, so the engine's licensed stream must be a prefix-consistent subsequence of
the no-spec greedy stream. We anchor each round by locating verify[0] in that stream.
"""
import pathlib
import re

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
    out = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            out = [int(x) for x in line.split("generated ids", 1)[1].split()]
    return out


rounds = parse("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_probe_raw.log")
gt = plain_ids("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_plain.log")

# --- anchor every round on the ground-truth stream ---------------------------
cursor = 0
for r in rounds:
    r["s"] = None
    if not r["cols"]:
        continue
    head = r["cols"][0]["verify"]
    for idx in range(cursor, len(gt)):
        if gt[idx] == head:
            r["s"] = idx
            cursor = idx
            break
    if r["s"] is None:
        cursor = 0
        for idx in range(len(gt)):
            if gt[idx] == head:
                r["s"] = idx
                break

anchored = [r for r in rounds if r["s"] is not None]
# the lagged `accepted` means round r's real outcome is round r+1's accepted field
for i, r in enumerate(rounds):
    r["own_accepted"] = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
    r["own_extent"] = rounds[i + 1]["in_extent"] if i + 1 < len(rounds) else None

print(f"rounds={len(rounds)}  anchored={len(anchored)}  gt={len(gt)}")
mism = [r for r in rounds if r["s"] is None]
print(f"rounds that could NOT be anchored on the gt stream: {len(mism)}")
print()

drafting = [r for r in anchored if r["in_extent"] > 0]
print(f"rounds with a draft block: {len(drafting)}")
print()

# --- does the engine's licensed stream stay on the gt stream? ----------------
jumps = [(r["s"], r["own_accepted"], r["own_extent"]) for r in drafting]
print("first 12 drafting rounds: gt_offset s, accepted(lagged-corrected), extent")
for s, a, e in jumps[:12]:
    print(f"   s={s:<4} accepted={a} extent={e}")
print()

W = max(len(r["cols"]) for r in rounds)
tgt_ok = [0] * W
drf_ok = [0] * W
agree = [0] * W
n = [0] * W
for r in drafting:
    for c in r["cols"]:
        j, s = c["col"], r["s"]
        t = s + j + 1
        if t >= len(gt) or c["draft"] == -1:
            continue
        n[j] += 1
        if c["argmax"] == gt[t]:
            tgt_ok[j] += 1
        if c["draft"] == gt[t]:
            drf_ok[j] += 1
        if c["argmax"] == c["draft"]:
            agree[j] += 1

print("col |   n | TARGET argmax right | DRAFT right | argmax==draft")
for j in range(W):
    if not n[j]:
        continue
    print(f"  {j} | {n[j]:>3} | {tgt_ok[j]:>3} ({100.0*tgt_ok[j]/n[j]:>5.1f}%)      "
          f"| {drf_ok[j]:>3} ({100.0*drf_ok[j]/n[j]:>5.1f}%)   | {agree[j]:>3}")
print()

acc = [r for r in drafting if (r["own_accepted"] or 0) >= 1]
print(f"rounds where the engine really accepted >=1 draft: {len(acc)}")
if acc:
    t1 = sum(1 for r in acc if len(r["cols"]) > 1
             and r["s"] + 2 < len(gt) and r["cols"][1]["argmax"] == gt[r["s"] + 2])
    d1 = sum(1 for r in acc if len(r["cols"]) > 1
             and r["s"] + 2 < len(gt) and r["cols"][1]["draft"] == gt[r["s"] + 2])
    print(f"   at column 1:  TARGET argmax right {t1}/{len(acc)}   "
          f"DRAFT right {d1}/{len(acc)}")
    print("   (reference dflash2 keeps ~82% of accepted prefixes at column 1)")
print()
print("=== per-round detail (draft blocks only, first 10) ===")
for r in drafting[:10]:
    s = r["s"]
    print(f"  s={s} accepted={r['own_accepted']} extent={r['own_extent']} "
          f"verify[0]={r['cols'][0]['verify']}")
    print(f"    gt[s+1..] : {gt[s+1:s+1+W]}")
    print(f"    argmax    : {[c['argmax'] for c in r['cols']]}")
    print(f"    draft     : {[c['draft'] for c in r['cols']]}")
    marks = []
    for c in r["cols"]:
        t = s + c["col"] + 1
        marks.append("T" if (t < len(gt) and c["argmax"] == gt[t]) else
                     ("b" if t < len(gt) and c["argmax"] != gt[t] else " "))
    print(f"    target_ok : {''.join(marks)}   (T=target argmax == truth)")
