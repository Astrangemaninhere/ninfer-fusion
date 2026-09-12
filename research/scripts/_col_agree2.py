#!/usr/bin/env python3
"""列 0 一致率（修正取数路径）：探针 dl/probe_now.log (新二进制) vs /home/user/bl_zh_plain.log (新二进制 plain)。"""
import pathlib, re
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
PROBE = D / "probe_now.log"
PLAIN = pathlib.Path("/home/user/bl_zh_plain.log")

ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

rounds, cur = [], None
for line in PROBE.read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        cur = {"accepted": int(m.group(7)), "count": int(m.group(8)), "cols": []}
        rounds.append(cur); continue
    m = COL.search(line)
    if m and cur is not None:
        cur["cols"].append({"col": int(m.group(1)), "verify": int(m.group(3)),
                            "draft": int(m.group(4)), "argmax": int(m.group(5))})
plain = plain_ids(PLAIN)
print(f"探针轮数={len(rounds)}  plain ids={len(plain)}")

# 锚定：探针每轮列 0 的 verify 就是当轮发布的 token；在 plain 序列里前向匹配
cursor = 0
for r in rounds:
    r["s"] = None
    if r["cols"]:
        head = r["cols"][0]["verify"]
        for i in range(cursor, len(plain)):
            if plain[i] == head:
                r["s"] = i; cursor = i; break
anchored = [r for r in rounds if r["s"] is not None]
for i, r in enumerate(rounds):
    r["own"] = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None

W = max((len(r["cols"]) for r in rounds), default=0)
n = [0]*W; ok = [0]*W; bad = {}
for r in anchored:
    if r["own"] is None: continue
    for c in r["cols"]:
        j = c["col"]; t = r["s"] + j + 1
        if j > max(0, r["own"]) or t >= len(plain) or c["verify"] == -1: continue
        n[j] += 1
        if c["argmax"] == plain[t]:
            ok[j] += 1
        elif j == 0:
            bad[(c["verify"], c["argmax"], plain[t])] = bad.get((c["verify"], c["argmax"], plain[t]), 0) + 1

print(f"可锚定轮={len(anchored)}/{len(rounds)}   列宽 W={W}")
print(" 列 | 干净样本 | 与 plain 一致 | 一致率")
tn = to = 0
for j in range(W):
    if n[j]:
        print(f" {j:>3} | {n[j]:>8} | {ok[j]:>12} | {100.0*ok[j]/n[j]:>6.1f}%")
        tn += n[j]; to += ok[j]
if tn:
    print(f" 计  | {tn:>8} | {to:>12} | {100.0*to/tn:>6.1f}%")
if bad:
    print("列0 错例 (verify, 探针argmax, plain) 前 6:")
    for k, v in sorted(bad.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   verify={k[0]} 探针argmax={k[1]} plain={k[2]}  x{v}")
acc = sum(r["accepted"] for r in rounds if r["accepted"] >= 0)
cnt = sum(r["count"] for r in rounds if r["count"] > 0)
print(f"接受率 = {acc}/{cnt} = {100.0*acc/cnt if cnt else 0:.2f}%")
