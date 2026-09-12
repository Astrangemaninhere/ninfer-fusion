#!/usr/bin/env python3
"""直接读探针原始行：检查 verify 各列的 pos 是否 = anchor + j，以及 verify/argmax/draft 的关系。"""
import pathlib, re
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

for tag in ("w8", "w2"):
    p = D / f"rd_{tag}.log"
    print("="*72)
    print(f"[{tag}] 前 4 轮原始内容")
    shown = 0
    for line in p.read_text(errors="replace").splitlines():
        if "df2dbg" not in line: continue
        if ROUND.search(line):
            if shown >= 4: break
            shown += 1
            print("  " + line.split("df2dbg")[-1].strip()[:130])
        elif COL.search(line):
            print("      " + line.split("df2dbg")[-1].strip()[:110])
