#!/usr/bin/env python3
"""目标行偏移扫描：现有 artifact 的草稿，究竟对齐到哪个位置？
引擎当前按 slot j -> 位置 anchor+j+1 对齐（一致率 78.3%）。
扫 offset ∈ {-1,0,+1,+2}（即 t = s+j+1+off），看哪个偏移让与 plain 的一致率最高。
若 +1 明显更高 => 头是按"晚一位"训的 => 引擎应当迁就它（零训练），而不是回头重训头。
"""
import pathlib, re
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

def parse(p):
    rounds, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m: cur = {"accepted": int(m.group(7)), "cols": []}; rounds.append(cur); continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "verify": int(m.group(3)), "argmax": int(m.group(5))})
    return rounds

plain = plain_ids(D / "rd_plain_ids.log")
rd = parse(D / "probe_now.log")
# 锚定（与之前一致）：用本轮 col0 的 verify 在 plain 里前向匹配
cur = 0
for r in rd:
    r["s"] = None
    if r["cols"]:
        head = r["cols"][0]["verify"]
        for i in range(cur, len(plain)):
            if plain[i] == head: r["s"] = i; cur = i; break
for i, r in enumerate(rd):
    r["own"] = rd[i+1]["accepted"] if i+1 < len(rd) else None

print("目标行偏移扫描（offset 相对引擎当前对齐 s+j+1）：")
print(" off | 干净样本 | 与 plain 一致 | 一致率")
for off in (-1, 0, 1, 2):
    n = o = 0
    for r in rd:
        if r["s"] is None or r["own"] is None: continue
        for c in r["cols"]:
            j = c["col"]; t = r["s"] + j + 1 + off
            if j > max(0, r["own"]) or t < 0 or t >= len(plain) or c["verify"] == -1: continue
            n += 1; o += (c["argmax"] == plain[t])
    tag = "  <- 引擎当前" if off == 0 else ""
    print(f" {off:+d} | {n:>8} | {o:>12} | {100.0*o/n if n else 0:>6.1f}%{tag}")
print()
print("另：草稿本身命中 verify 的 argmax（判定接受率的口径，与偏移无关）")
n = o = 0
for r in rd:
    if r["s"] is None: continue
    for c in r["cols"]:
        if c["argmax"] == -1: continue
        n += 1; o += (c["argmax"] == c["verify"])
print(f"  草稿==argmax(同位置) : {o}/{n} = {100.0*o/n if n else 0:.1f}%")
