#!/usr/bin/env python3
"""漂移累积轴：修掉 +1 偏移后重算（第 0 个生成 token 来自 prefill，不属任何 decode 轮）。"""
import pathlib
import re

D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
round_re = re.compile(r"row=\d+ anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                      r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
col_re = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")


def parse(p):
    out, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        m = round_re.search(line)
        if m:
            cur = {"accepted": int(m.group(6)), "count": int(m.group(7)), "cols": []}
            out.append(cur)
            continue
        m = col_re.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return out


def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []


plain = plain_ids(D / "ax_plain.log")
print(f"plain n={len(plain)}   plain[0..4]={plain[:5]}")
for k in (1, 7):
    rounds = parse(D / f"ax_k{k}.probe")
    for i, r in enumerate(rounds):
        r["own"] = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
    idx = 1                      # prefill 已产出 plain[0]
    first = None
    for n, r in enumerate(rounds):
        if r["own"] is None or not r["cols"]:
            continue
        a = max(0, r["own"])
        toks = [c["draft"] for c in r["cols"][:a]]
        toks.append(r["cols"][min(a, len(r["cols"]) - 1)]["argmax"])
        for t in toks:
            if idx >= len(plain):
                break
            if t != plain[idx] and first is None:
                first = (n + 1, idx)
            idx += 1
    print(f"K={k}: rounds={len(rounds)} 生成={idx}  "
          f"首次偏离 = 第 {first[0] if first else '-'} 轮 / 第 {first[1] if first else '-'} 个生成 token")
    if first:
        n, i = first
        print(f"      plain[{i}]={plain[i]}  该轮以后 spec 侧开始分叉")
