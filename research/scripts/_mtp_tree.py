#!/usr/bin/env python3
"""MTP 树的收益量化（用已有实测 α 剖面，GPU-free）。

链式： EAL_chain(d) = 1 + Σ_{j<=d} Π_{i<=j} α_i
树 b ： EAL_tree(b,d) ≈ 1 + Σ_{j<=d} Π_{i<=j} (1 - (1-α_i)^b)
效率 ： η = (EAL-1)/verify_slots ，slots_tree = 1 + b*d（上限 16，见 round_state.h:18）
成本证据：MVP-0 实测列数 8→16 只慢 4.2% ⇒ verify 槽位近乎免费；草稿侧 MTP 每步仅 1 层前向。
"""
import pathlib
import re

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")

PROFILES = {
    "mtp_zh(32,14,4)/45轮": ([32, 14, 4], 45),
    "mtp_en(28,18,11)": ([28, 18, 11], None),
    "mtp_code(24,22,21)": ([24, 22, 21], None),
    "dflash2_zh(20,3,0,0,0,0,0)": ([20, 3, 0, 0, 0, 0, 0], 71),
    "dflash2_en(22,16,10,7,3,2,2)": ([22, 16, 10, 7, 3, 2, 2], 33),
}


def alpha_from(counts, rounds):
    a, prev = [], rounds if rounds else max(counts[0], 1)
    for c in counts:
        a.append(c / prev if prev else 0.0)
        prev = c
    return a


def eal_chain(a):
    p, tot = 1.0, 0.0
    for x in a:
        p *= x
        tot += p
    return 1.0 + tot


def eal_tree(a, b):
    p, tot = 1.0, 0.0
    for x in a:
        p *= (1.0 - (1.0 - x) ** b)
        tot += p
    return 1.0 + tot


print(f"{'剖面':<30} {'α 逐位':<34} {'链':<7} {'树b=2':<8} {'树b=4':<8} {'列数(b2/b4)'}")
for name, (counts, rounds) in PROFILES.items():
    a = alpha_from(counts, rounds)
    ec, e2, e4 = eal_chain(a), eal_tree(a, 2), eal_tree(a, 4)
    d = len(a)
    print(f"{name:<30} {str([round(x,3) for x in a]):<34} "
          f"{ec:<7.2f} {e2:<8.2f} {e4:<8.2f} {1+2*d}/{1+4*d}")

print()
print("=== 注：MTP 的 --draft-tokens 上限为 5，故深度 d<=5；verify 列上限 16 ===")
print("  b=2,d=5 -> 11 列;  b=4,d=3 -> 13 列;  b=4,d=5 -> 21 列(超限)")
print()
print("=== 成本侧证据（本会话实测）===")
print("  列数 8 -> 16：tok/s 44.39 -> 42.59（-4.2%）⇒ verify 槽位近乎免费")
print("  MTP 草稿每步 = 1 层头前向；d=3,b=4 的树共 1+4+16=21 次 1 层前向（链为 3 次）")
