#!/usr/bin/env python3
"""逐步体检（修正版）：verify 每列的 argmax 应等于 plain 流在同一位置的 token。

列 j 在绝对位置 pos 上预测的是位置 pos+1 的 token；plain 流的第 i 个 token 在绝对位置
prompt_len + i。故比较：verify.argmax[pos]  ==  plain[pos - prompt_len + 1]。
第一个不等的列就是 target logits 的漂移点。
"""

import pathlib
import re

D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
round_re = re.compile(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                      r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
col_re = re.compile(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) "
                    r"argmax=(-?\d+)")


def rounds_from(path):
    out, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = round_re.search(line)
        if m:
            cur = {"anchor": int(m.group(2)), "frontier": int(m.group(3)),
                   "valid": int(m.group(4)), "in_extent": int(m.group(5)),
                   "accepted": int(m.group(7)), "count": int(m.group(8)), "cols": []}
            out.append(cur)
            continue
        m = col_re.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return out


def plain_ids(path):
    ids = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            ids = [int(x) for x in line.split("generated ids", 1)[1].split()]
    return ids


def prompt_len(*paths):
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if "prompt tokens" in line:
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return None
    return None


plen = prompt_len(D / "sw_stdout.log", D / "sw_probe.log")
plain = plain_ids(D / "sw_plain.log")
rounds = rounds_from(D / "sw_probe.log")
print(f"prompt_len={plen}  plain={len(plain)}  rounds={len(rounds)}")
if not plen or not plain:
    print("缺 prompt_len 或 plain；检查 dl/sw_stdout.log 与 dl/sw_plain.log")
    raise SystemExit(1)

# 每个绝对位置第一次被 verify 覆盖时的信息
seen = {}
for r in rounds:
    for c in r["cols"]:
        p = c["pos"]
        if p not in seen:
            seen[p] = dict(c, in_extent=r["in_extent"], accepted=r["accepted"],
                           count=r["count"], frontier=r["frontier"])
print(f"被 verify 覆盖的绝对位置数 = {len(seen)}")
print()

checked = bad = 0
first_bad = None
col0_rounds = 0
col0_bad = 0
first_col0_bad = None
for p in sorted(seen):
    tgt = p - plen + 1
    if tgt < 0 or tgt >= len(plain):
        continue
    checked += 1
    if seen[p]["argmax"] != plain[tgt]:
        bad += 1
        if first_bad is None:
            first_bad = (p, tgt)
    # 只有第 0 列（输入=真 anchor、上下文=真实已接受前缀）必须等于 plain
    if seen[p]["col"] == 0:
        col0_rounds += 1
        if seen[p]["argmax"] != plain[tgt]:
            col0_bad += 1
            if first_col0_bad is None:
                first_col0_bad = (p, tgt)
print(f"全部覆盖位置 {checked}：与 plain 不一致 {bad} 个（含 col>=1 的正常污染）")
print(f"**仅第 0 列** {col0_rounds} 个：与 plain 不一致 {col0_bad} 个")
if first_col0_bad:
    p, tgt = first_col0_bad
    c = seen[p]
    print(f"   首个 col0 漂移：pos={p}（生成序 {tgt}）投喂 token={c['verify']} "
          f"verify.argmax={c['argmax']} plain={plain[tgt]} "
          f"in_extent={c['in_extent']} accepted={c['accepted']}")
else:
    print("   ⇒ col0 全部与 plain 一致：verify 在第 0 列（真 anchor 列）与 plain 等价")
print()

if first_bad:
    p, tgt = first_bad
    c = seen[p]
    print(f"** 首个漂移列 ** 绝对位置 pos={p}（生成序 {tgt}）col={c['col']} "
          f"投喂 token={c['verify']} verify.argmax={c['argmax']} plain={plain[tgt]} "
          f"in_extent={c['in_extent']} accepted={c['accepted']} frontier={c['frontier']}")
    lo = max(0, tgt - 5)
    print(f"   plain[{lo}:{tgt+5}] = {plain[lo:tgt+5]}")
    # 该位置前后几个被覆盖的列，看是"整块偏移"还是"单列不同"
    window = [(q, seen[q]['verify'], seen[q]['argmax'],
               plain[q - plen + 1] if 0 <= q - plen + 1 < len(plain) else None)
              for q in sorted(seen) if abs(q - p) <= 4]
    print("   邻近列  pos / verify / argmax / plain-at-pos:")
    for q, v, a, pl in window:
        flag = "" if a == pl else "   <-- 不等"
        print(f"     {q:>5} {v:>8} {a:>8} {pl if pl is not None else '-':>8}{flag}")
    print()
    print("   该轮全部列（verify / argmax）:")
    for r in rounds:
        if any(c["pos"] == p for c in r["cols"]):
            print("     verify:", [c["verify"] for c in r["cols"]])
            print("     argmax:", [c["argmax"] for c in r["cols"]])
            break
else:
    print("所有可比较位置一致 ⇒ target logits 无偏差，问题在 accept/记账侧")
