#!/usr/bin/env python3
#!/usr/bin/env python3
# ============================================================================
# WARNING / 声明（w-fit3 于 2026-09-14 加）—— 请先读完这段再用这个文件。
#
#   这是一份【已分叉的历史实现】，它的阶梯（TIERS / PENALTY / RK4V4_LAYER_LIMIT）与权威
#   实现 src/product/kv_bit_budget.h 的 kKvBitBudgetTiers 已经【不一致】，而只有后者
#   是权威的。它被 port 进来一次（commit f5c4def）之后再也没有更新过。
#
#   分叉在哪（w-fit3 实测；权威侧 = C++ 头 src/product/kv_bit_budget.h）：
#       tier   bits(C++/权威)   bits(本文件)     penalty(C++)   penalty(本文件)
#       fp8        8.50          8.03 ← 漂        0.03           0.03
#       rk4v4         4.25          4.06 ← 漂        0.08           0.08
#       iso4e       4.50          3.00 ← 漂        2.00           0.50 ← 漂
#       rk4v4 层数上限：C++ = 10（实测支持），本文件 RK4V4_LAYER_LIMIT = 8
#
#   同一个预算在两套阶梯下会给出【不同的部署表】（逐预算对照表见
#   /home/user/scratch/w_fit3/REPORT.md 的「两套阶梯的逐预算对照」一节）。
#
#   【不要用它做决策】：不要拿它的 stdout 当部署表，不要把它 TIERS/PENALTY 里的数字
#   当引擎代价。引擎侧唯一权威是 src/product/kv_bit_budget.h，而环境里唯一的消费者是
#   那个头（w-fit3 核过：tools/ 与 tests/ 下的 kv_bit_budget 命中全是 C++ 符号，没有
#   任何东西 import 或调用本文件）。它保留在树里是因为删它会去掉一份可对照的历史记
#   录；它需要这段声明，是因为它一展开就会被当成第二套真相。
#
#   历史事实（w-fit3 查的 path-level git log）：本文件路径只有过一个提交
#   f5c4def "tools: port the additive archkit/FlashNext tooling and research docs
#   from the remote line" —— 它从未被删除过。失真的是文档：
#   src/product/kv_bit_budget.h:3-5 曾写着 "is no longer in the tree"。
# ============================================================================
"""kv_bit_budget.py — given a KV bit budget (fractional), emit the best per-layer KV table.

Model
  Each full-attention layer picks one KV tier. A tier costs a fixed number of bits per KV
  element (format math, see the engine's KV storage enum) and carries a quality penalty
  (measured/prior). For a target average bit budget B and L layers the allocator solves
      minimize  sum(penalty(tier_i))   s.t.  sum(bits(tier_i)) <= B * L
  exactly with a small DP (bits discretized to 0.01).

Tier bit costs (bits/element, K+V averaged):
  bf16 16.0 | fp8 8.03 | int8 8.25 | nvfp4 4.50 | rk4v4 4.06 | iso4e 3.00

Quality penalties (needle-sweep prior, lower is better; refine with kv_matrix_v3 results):
  bf16 0.00 | int8 0.02 | fp8 0.03 | rk4v4 0.08 | nvfp4 0.30 | iso4e 0.50

Usage:
  kv_bit_budget.py --layers 8 --bits 4.5
  kv_bit_budget.py --layers 8 --bits 3.5 4.0 4.5 5 6 8 12 16
"""
from __future__ import annotations

import argparse

TIERS = {
    "bf16": 16.00,
    "fp8": 8.03,
    "int8": 8.25,
    "nvfp4": 4.50,
    "rk4v4": 4.06,
    "iso4e": 3.00,
}
PENALTY = {
    "bf16": 0.00,
    "int8": 0.02,
    "fp8": 0.03,
    "rk4v4": 0.08,
    "nvfp4": 0.30,
    "iso4e": 0.50,
}
ORDER = ["bf16", "int8", "fp8", "nvfp4", "rk4v4", "iso4e"]  # cost ladder for the DP
# Packing order: rk4v4 first so the verified leading layers get it (the DP only counts tiers).
PACK_ORDER = ["rk4v4", "bf16", "int8", "fp8", "nvfp4", "iso4e"]
# rk4v4 is only verified on the first N full-attention layers (see _TODO.md 95: layers 8-15 of
# qwen3.8-27b return garbage with rk4v4, while iso4e/int8 are fine there).
RK4V4_LAYER_LIMIT = 8


def solve(layers: int, budget_bits: float, rk4v4_limit: int = RK4V4_LAYER_LIMIT
          ) -> tuple[dict[str, int], float, float]:
    """Exact DP: returns (tier -> layer count, achieved bits/element, total penalty)."""
    scale = 100
    capacity = int(round(budget_bits * scale)) * layers
    # dp[(layer, bits)] = (penalty, counts) -- penalty minimised
    dp: dict[tuple[int, int], tuple[float, dict[str, int]]] = {(0, 0): (0.0, {})}
    for _ in range(layers):
        nxt: dict[tuple[int, int], tuple[float, dict[str, int]]] = {}
        for (used_layers, bits), (penalty, counts) in dp.items():
            for tier in ORDER:
                if tier == "rk4v4" and counts.get("rk4v4", 0) >= rk4v4_limit:
                    continue
                cost = int(round(TIERS[tier] * scale))
                new_bits = bits + cost
                if new_bits > capacity:
                    continue
                new_counts = dict(counts)
                new_counts[tier] = new_counts.get(tier, 0) + 1
                key = (used_layers + 1, new_bits)
                candidate = (penalty + PENALTY[tier], new_counts)
                if key not in nxt or candidate[0] < nxt[key][0]:
                    nxt[key] = candidate
        dp = nxt
    best = None
    for (used_layers, bits), (penalty, counts) in dp.items():
        if used_layers != layers:
            continue
        if best is None or penalty < best[0]:
            best = (penalty, bits, counts)
    assert best is not None, "no feasible allocation"
    penalty, bits, counts = best
    return counts, bits / (layers * scale), penalty


def to_spec(counts: dict[str, int], layers: int) -> str:
    """Pack the tier counts into the --kv-layer-storage grammar (ranges per tier)."""
    slots: list[str | None] = [None] * layers
    cursor = 0
    for tier in PACK_ORDER:
        for _ in range(counts.get(tier, 0)):
            slots[cursor] = tier
            cursor += 1
    parts: list[str] = []
    begin = 0
    while begin < layers:
        tier = slots[begin]
        end = begin
        while end + 1 < layers and slots[end + 1] == tier:
            end += 1
        label = f"{begin}:{tier}" if begin == end else f"{begin}-{end}:{tier}"
        parts.append(label)
        begin = end + 1
    return ",".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8,
                    help="full-attention layers in the model (qwen3.8-27b: 8)")
    ap.add_argument("--bits", type=float, nargs="+", required=True,
                    help="target average bits per KV element (fractional allowed)")
    ap.add_argument("--rk4v4-layers", type=int, default=RK4V4_LAYER_LIMIT,
                    help="how many leading layers may use rk4v4 (verified range only)")
    args = ap.parse_args()
    print(f"# layers={args.layers} rk4v4_limit={args.rk4v4_layers} tiers={TIERS}")
    for target in args.bits:
        counts, achieved, penalty = solve(args.layers, target, args.rk4v4_layers)
        spec = to_spec(counts, args.layers)
        mix = "+".join(f"{tier}x{count}" for tier, count in sorted(counts.items()))
        print(f"bits={target:5.2f} -> achieved={achieved:5.2f} penalty={penalty:5.2f} "
              f"mix={mix:28s} --kv-layer-storage {spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
