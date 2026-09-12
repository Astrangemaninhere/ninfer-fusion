#!/usr/bin/env python3
"""当前二进制逐列一致性（纯分析；探针已在 dl/probe_now.log，plain 在 /home/user/bl_zh_plain.log）。"""
import pathlib
import re

PROBE = "/mnt/c/Users/User/Documents/ziqinzhang/dl/probe_now.log"
PLAIN = "/home/user/bl_zh_plain.log"
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")


def parse(path):
    rounds, cur = [], None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            cur = {"accepted": int(m.group(7)), "cols": []}
            rounds.append(cur)
            continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "argmax": int(m.group(5))})
    return rounds


def plain_ids(path):
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []


rounds = parse(PROBE)
plain = plain_ids(PLAIN)
print(f"轮数={len(rounds)} plain={len(plain)}")
cursor = 0
for r in rounds:
    r["s"] = None
    if r["cols"]:
        head = r["cols"][0]["verify"]
        for idx in range(cursor, len(plain)):
            if plain[idx] == head:
                r["s"] = idx
                cursor = idx
                break
anchored = [r for r in rounds if r["s"] is not None]
for i, r in enumerate(rounds):
    r["own"] = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
W = max((len(r["cols"]) for r in rounds), default=0)
n = [0] * W
ok = [0] * W
for r in anchored:
    if r["own"] is None:
        continue
    for c in r["cols"]:
        j = c["col"]
        t = r["s"] + j + 1
        if j > max(0, r["own"]) or t >= len(plain) or c["verify"] == -1:
            continue
        n[j] += 1
        ok[j] += (c["argmax"] == plain[t])
print(f"可锚定轮={len(anchored)}")
print(" 列 | 干净样本 | 与 plain 一致 | 一致率")
tn = to = 0
for j in range(W):
    if n[j]:
        print(f" {j:>3} | {n[j]:>8} | {ok[j]:>12} | {100.0*ok[j]/n[j]:>6.1f}%")
        tn += n[j]
        to += ok[j]
if tn:
    print(f" 计  | {tn:>8} | {to:>12} | {100.0*to/tn:>6.1f}%")
print("（历史对比：F4/① 之前 列0 17/20=85%、列1 6/8=75%）")
