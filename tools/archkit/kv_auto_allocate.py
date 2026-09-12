#!/usr/bin/env python3
"""kv_auto_allocate.py — user-set K/V quant grades -> optimal per-layer KV storage spec.

Contract (user-facing): you pick a K grade and a V grade; the allocator decides
how many layers run the cheap tier and how many run the precise tier, maximizing
KV capacity (bytes/token -> prompt headroom) while keeping needle accuracy
acceptable (calibrated coverage limits, §51/§52).

Engine semantics used (decoder_state.cpp):
  layer=nvfp4  -> K=NVFP4(E2M1+e4m3) + V=ISO3 (hardwired pair, nibble planes)
  layer=e8     -> K=E8 2-bit lattice + V=E8 (cheapest bytes/token)
  layer=bf16/i8/fp8 -> K=V=same
So "grades" map to tier pairs; the allocator chooses the cheapest tier whose
calibrated coverage stays within the needle-safe limit.

Usage:
  kv_auto_allocate.py --model muse_glimmer_30b --k-grade auto --v-grade auto \
      --target-context 65536 --layers 16
    --k-grade/--v-grade: nvfp4|e8|iso3|int8|fp8|bf16|auto (auto = cheapest safe)
  Output: the --kv-layer-storage spec + bytes/token math + expected capacity.
"""
from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Tier table: bytes per token per layer for GQA Muse geometry (32 kv heads,
# head_dim 128, K+V). Values = measured plane math; order by cost.
# Format pair: (label, k_dtype, v_dtype, bytes_per_elem_kv)
#   nvfp4 pair: K 4-bit codes (0.5B) + V ISO3 nibble (0.5B) + scales(~2/16)*2*2B
#   e8 pair:    K 2-bit-in-int8 lattice slot (1B) + V 1B, g64 scales negligible
#   i8 pair:    1B + 1B + g64 fp16 scales
#   fp8 pair:   1B + 1B + row256 scales
#   bf16:       2B + 2B
TIERS = {
    "e8":    {"label": "e8",    "k": "e8",    "v": "e8",    "bytes_tok_layer": 32 * 128 * 2 * 1.06},
    "nvfp4": {"label": "nvfp4", "k": "nvfp4", "v": "iso3",  "bytes_tok_layer": 32 * 128 * 2 * 0.56},
    "int8":  {"label": "int8",  "k": "int8",  "v": "int8",  "bytes_tok_layer": 32 * 128 * 2 * 1.06},
    "fp8":   {"label": "fp8",   "k": "fp8",   "v": "fp8",   "bytes_tok_layer": 32 * 128 * 2 * 1.03},
    "bf16":  {"label": "bf16",  "k": "bf16",  "v": "bf16",  "bytes_tok_layer": 32 * 128 * 2 * 2.0},
}

# Calibrated needle-safe coverage limits (fraction of layers allowed at the
# cheap tier before the quality cliff). Source: kv_calibrate sweeps (§51/§52:
# E8 coverage <= 12/16 = 75% at 57K needle PASS; 14/16 FAIL). Extend per model
# by running kv_calibrate.py with the combo's spec.
COVERAGE_LIMITS = {
    ("muse_glimmer_30b", "e8"): 12,
}

DEFAULT_MODEL = "muse_glimmer_30b"


def allocate(model: str, k_grade: str, v_grade: str, layers: int) -> dict:
    limit = COVERAGE_LIMITS.get((model, "e8"))
    if limit is None:
        # conservative fallback: 75% coverage like the Muse calibration
        limit = int(layers * 0.75)

    # cheapest tier first; "auto" grades mean "cheapest acceptable"
    cheap = "e8"
    if k_grade not in ("auto", "", None):
        # user pinned K; if it's not a cheap tier there is nothing to allocate
        pass
    precise = "nvfp4"

    n_cheap = limit
    if k_grade in ("nvfp4", "int8", "fp8", "bf16") and v_grade in ("auto", "iso3", "nvfp4", "", None):
        # user pinned K to a precise tier: no cheap coverage, all precise
        n_cheap = 0
        precise = k_grade

    spec_layers = []
    for layer in range(layers):
        spec_layers.append((layer, cheap if layer < n_cheap else precise))

    # compact into ranges
    ranges = []
    for layer, tier in spec_layers:
        if ranges and ranges[-1][2] == tier and layer == ranges[-1][1] + 1:
            ranges[-1][1] = layer
        else:
            ranges.append([layer, layer, tier])
    spec = ",".join(
        f"{a}:{t}" if a == b else f"{a}-{b}:{t}" for a, b, t in ranges
    )

    bytes_tok = sum(
        TIERS[t]["bytes_tok_layer"] for _, t in spec_layers
    )
    return {
        "model": model,
        "k_grade": k_grade,
        "v_grade": v_grade,
        "cheap_tier": cheap,
        "precise_tier": precise,
        "cheap_layers": n_cheap,
        "precise_layers": layers - n_cheap,
        "coverage": f"{n_cheap}/{layers} = {n_cheap / layers:.0%}",
        "safe_limit_basis": COVERAGE_LIMITS.get((model, "e8"), "fallback 75%"),
        "bytes_per_token_total": round(bytes_tok, 1),
        "kv_layer_storage": spec,
        "cmdline": f"--kv-layer-storage {spec}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k-grade", default="auto",
                    help="nvfp4|e8|int8|fp8|bf16|auto (auto = cheapest safe)")
    ap.add_argument("--v-grade", default="auto",
                    help="iso3|nvfp4|auto (engine pairs V with the layer tier today)")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--target-context", type=int, default=0,
                    help="report capacity at this context (informational)")
    args = ap.parse_args()

    result = allocate(args.model, args.k_grade, args.v_grade, args.layers)
    if args.target_context:
        need_gib = result["bytes_per_token_total"] * args.target_context / 2**30
        result["target_context"] = args.target_context
        result["kv_gib_at_target"] = round(need_gib, 2)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
