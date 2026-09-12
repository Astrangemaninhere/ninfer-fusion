#!/usr/bin/env python3
"""逐列等价性表：verify 块的第 j 列在"上下文干净"（前 j 列全部被接受、即投喂的就是真 token）
时，其 argmax 必须等于 plain 流在同一位置的 token。

- j=0：输入=真 anchor，上下文=真实已接受前缀（永远干净）
- j>=1：只有当第 0..j-1 列都被接受时，第 j 列的输入才是真 token、上下文才是真前缀

按列统计 "干净样本数 / 与 plain 不一致数"，并打印首个不一致的细节。
"""

import pathlib
import re

D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
round_re = re.compile(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                      r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
col_re = re.compile(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) "
                    r"argmax=(-?\d+)")


def parse_rounds(path):
    out, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = round_re.search(line)
        if m:
            cur = {"anchor": int(m.group(2)), "frontier": int(m.group(3)),
                   "in_extent": int(m.group(5)), "accepted": int(m.group(7)),
                   "count": int(m.group(8)), "cols": []}
            out.append(cur)
            continue
        m = col_re.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return out


def read_ids(path):
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []


def read_prompt_len(*paths):
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if "prompt tokens" in line:
                return int(line.split()[-1])
    return None


plen = read_prompt_len(D / "sw_stdout.log", D / "sw_probe.log")
plain = read_ids(D / "sw_plain.log")
rounds = parse_rounds(D / "sw_probe.log")
print(f"prompt_len={plen}  plain={len(plain)}  rounds={len(rounds)}")

# 探针的 accepted/count/in_extent 滞后一轮：本轮真实结果在下一行，先做滞后校正。
for i, r in enumerate(rounds):
    nxt = rounds[i + 1] if i + 1 < len(rounds) else None
    r["own_accepted"] = nxt["accepted"] if nxt else None
    r["own_extent"] = nxt["in_extent"] if nxt else None

# 每列统计：干净样本 / 不一致
stat = {}
first_bad = None
for r in rounds:
    cols = r["cols"]
    if not cols or r["own_accepted"] is None:
        continue
    # 该轮真实"干净"的列数 = 本轮接受的 draft 数 + 1（第 0..own_accepted 列）
    clean_upto = r["own_accepted"] + 1
    for c in cols:
        j = c["col"]
        if j >= clean_upto:
            continue                      # 该列输入是草稿，上下文被污染：不计入等价性
        tgt = c["pos"] - plen + 1
        if tgt < 0 or tgt >= len(plain):
            continue
        n, bad = stat.get(j, (0, 0))
        ok = (c["argmax"] == plain[tgt])
        stat[j] = (n + 1, bad + (0 if ok else 1))
        if not ok and first_bad is None:
            first_bad = (j, c["pos"], tgt, c["verify"], c["argmax"], plain[tgt],
                         r["own_extent"], r["own_accepted"])

print()
print(" 列 | 干净样本 | 与 plain 不一致 | 一致率")
for j in sorted(stat):
    n, bad = stat[j]
    print(f" {j:>3} | {n:>8} | {bad:>14} | {(100.0*(n-bad)/n if n else 0):>6.1f}%")
tot_n = sum(n for n, _ in stat.values())
tot_b = sum(b for _, b in stat.values())
print(f" 合计 | {tot_n:>8} | {tot_b:>14} | {(100.0*(tot_n-tot_b)/tot_n if tot_n else 0):>6.1f}%")
print()
if first_bad:
    j, pos, tgt, vt, am, pl, ext, acc = first_bad
    print(f"** 首个干净上下文下的不一致 ** col={j} pos={pos} 投喂={vt} "
          f"verify.argmax={am} plain={pl} in_extent={ext} accepted={acc}")
else:
    print("⇒ 所有干净上下文列都与 plain 一致：块 verify 与顺序解码等价")
