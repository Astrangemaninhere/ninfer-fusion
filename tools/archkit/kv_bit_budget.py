#!/usr/bin/env python3
"""kv_bit_budget.py — given a KV bit budget (fractional), emit the best per-layer KV table.

Model
  Each full-attention layer picks one KV tier. A tier costs a fixed number of bits per KV
  element (format math, see the engine's KV storage enum) and carries a quality penalty
  (measured/prior). For a target average bit budget B and L layers the allocator solves
      minimize  sum(penalty(tier_i))   s.t.  sum(bits(tier_i)) <= B * L
  exactly with a small DP (bits discretized to 0.01).

Tier bit costs (bits/element, K+V averaged):
  bf16 16.0 | fp8 8.03 | int8 8.25 | nvfp4 4.50 | e8 4.06 | iso3 3.00

Quality penalties (needle-sweep prior, lower is better; refine with kv_matrix_v3 results):
  bf16 0.00 | int8 0.02 | fp8 0.03 | e8 0.08 | nvfp4 0.30 | iso3 0.50

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
    "e8": 4.06,
    "iso3": 3.00,
}
PENALTY = {
    "bf16": 0.00,
    "int8": 0.02,
    "fp8": 0.03,
    "e8": 0.08,
    "nvfp4": 0.30,
    "iso3": 0.50,
}
ORDER = ["bf16", "int8", "fp8", "nvfp4", "e8", "iso3"]  # cost ladder for the DP
# Packing order: e8 first so the verified leading layers get it (the DP only counts tiers).
PACK_ORDER = ["e8", "bf16", "int8", "fp8", "nvfp4", "iso3"]
# e8 is only verified on the first N full-attention layers (see _TODO.md 95: layers 8-15 of
# qwen3.8-27b return garbage with e8, while iso3/int8 are fine there).
E8_LAYER_LIMIT = 8


def solve(layers: int, budget_bits: float, e8_limit: int = E8_LAYER_LIMIT
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
                if tier == "e8" and counts.get("e8", 0) >= e8_limit:
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
    ap.add_argument("--e8-layers", type=int, default=E8_LAYER_LIMIT,
                    help="how many leading layers may use e8 (verified range only)")
    args = ap.parse_args()
    print(f"# layers={args.layers} e8_limit={args.e8_layers} tiers={TIERS}")
    for target in args.bits:
        counts, achieved, penalty = solve(args.layers, target, args.e8_layers)
        spec = to_spec(counts, args.layers)
        mix = "+".join(f"{tier}x{count}" for tier, count in sorted(counts.items()))
        print(f"bits={target:5.2f} -> achieved={achieved:5.2f} penalty={penalty:5.2f} "
              f"mix={mix:28s} --kv-layer-storage {spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
