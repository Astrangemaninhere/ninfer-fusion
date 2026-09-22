#!/usr/bin/env python
# ===========================================================================
# WARNING / STATUS (2026-09-15, KV-ADAPT-LAND): FORKED, DO NOT USE FOR DECISIONS
# ===========================================================================
# This file is kept as the historical record of an approach, NOT as a live
# decision-maker. Four named reasons, each independently checkable in this file:
#
#   1. ITS INPUT DOES NOT EXIST. It consumes .kvc frames produced by
#      `ninfer-cli --kv-calib-dir DIR`, and that flag is not implemented: a
#      per-file grep for "kv-calib-dir" over apps/cli/options.cpp (0), 
#      apps/cli/main.cpp (0), src/serve/serve_options.cpp (0),
#      src/serve/serve_options.h (0), src/product/kv_options.h (0) and
#      apps/perplexity/main.cpp (0) returns nothing, and no NINFERKVCAL1 writer
#      exists. So the program's only entry point cannot be reached.
#   2. ITS DECISION SPACE HAS NO rk4v4. The upgrade tuple at the greedy loop is
#      ("fp8", "int8", "bf16"). cost() has no rk4v4 branch either, so an rk4v4 layer
#      falls through to `return kv_heads * 4 * head_dim`, i.e. it is priced as
#      bf16 (16 b/el) when its real plane geometry is 4.25 b/el -- a 3.76x
#      overstatement. rk4v4 is the format the shipped table actually uses.
#   3. ITS BASE IS nvfp4 AND IT CAN ONLY UPGRADE, so it cannot represent the
#      shipped table at all: that table is nvfp4 WITH HOLES punched in it where
#      rk4v4 sits. `base = n_layers * cost("nvfp4")` plus monotone upgrades describes
#      a lattice that does not contain the shipped point.
#   4. IT IS GREEDY, NOT OPTIMAL. The candidate order in the upgrade loop is the
#      OPPOSITE of the cost order (int8 2112 B < fp8 2176 B per the cost()
#      planes), so fp8 is tried first; the KV-PERLAYER line measured a case where
#      the same 3 layers cost 192 B (1.04%) MORE under the greedy order than
#      under an int8-first order.
#
# THE LIVE REPLACEMENT is src/product/kv_adapt_solver.h: an exact multiple-choice
# knapsack over (layer, bytes) with a per-layer allowed set F_l, whose objective
# sum_l W[l][T(l)] carries the layer index that this file's composite score
# cannot. Its option family is built from kv_bit_budget.h's plane geometry, and
# its measurement schema is tools/archkit/kv_tier_matrix.py's JSON.
#
# If you DO re-enable this, add an rk4v4 column and a (K, V) pair per column first:
# the recorded failure mode of the previous generation was an E2M1-V ablation
# being compared against a deployed ISO4E-V layer, a 4.24x-6.56x NMSE error that
# a tier NAME cannot reveal because nvfp4's V plane is ISO4E.
# ===========================================================================
# Offline KV dynamic-precision calibration for NInfer Qwen3.6-family artifacts.
#
# Consumes the .kvc frames produced by `ninfer-cli --kv-calib-dir DIR` (exact
# post-RoPE K and V per full-attention layer and prefill chunk) and produces a
# static per-layer dtype table in the format accepted by `--kv-layer-storage`.
#
# Implemented metrics (the four requested techniques, fused into one decision):
#   MixKVQ      K/V error asymmetry weighting (K weighted above V).
#   TriAxialKV  per-layer, per-head, per-dimension-group outlier scores.
#   ARKV        effective rank (spectral spread) of K and V per head.
#   KVTuner     greedy sensitivity ranking under an explicit memory budget.
#
# The decision space is per-layer same-dtype storage (bf16 | int8 | nvfp4);
# the runtime's layer_kv_dtypes table consumes the selected map.
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<16s6I2i4I")
MAGIC = b"NINFERKVCAL1\x00\x00\x00\x00"

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E2M1_EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float64)


def e4m3fn(x: float) -> float:
    """Mimic the runtime gqa_kv_nvfp4_fp32_to_e4m3 (positive scale path)."""
    if not (x > 0.0):
        return 0.0
    b = struct.unpack("<I", np.float32(x).tobytes())[0]
    exp = int((b >> 23) & 0xFF) - 127 + 7
    if exp >= 15:
        return 480.0  # 核 code 0x7f 的解码值（ldexpf(1.875, 8)）
    if exp <= 0:
        # 核 src/ops/kernel/gqa_attention_kv_nvfp4.cuh:79-85 逐字：
        # "E4M3FN denormals decode as mantissa / 512 (mantissa * 2^-9), so the
        #  encoder must quantize x * 512, not x * 64."
        # roundf() 是 ties-away-from-zero（roundf(0.5)=1），Python round() 是
        # ties-to-even（round(0.5)=0）。同一个核里已经因这一点在 std::nearbyint
        # 处被骗过一次（kv_bit_budget.h:390-397）。
        mant = int(math.floor(x * 512.0 + 0.5))
        if mant <= 0:
            return 0.0
        if mant >= 8:
            # 编码写出的是 code 0x08，而 gqa_kv_nvfp4_e4m3_to_f32 对它的读数是
            # ldexpf(1.0, 1-7) = 2**-6 = 0.015625；原文的 1.0 是 64 倍错。
            return 2 ** -6
        return mant / 512.0
    mant = (b >> 20) & 0x7
    guard = (b >> 19) & 1
    sticky = b & 0x7FFFF
    if guard and (sticky or (mant & 1)):
        mant += 1
        if mant > 7:
            mant = 0
            exp += 1
            if exp >= 15:
                return 480.0
    return (1.0 + mant / 8.0) * 2 ** (exp - 7)


def quantize_e2m1_group16(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # groups along the last axis (head dim), 16 values each.
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 6.0, 2.0**-9)
    vq = np.array([e4m3fn(float(v)) for v in scale.ravel()], dtype=np.float64).reshape(scale.shape)
    q = np.abs(groups) / vq
    codes = np.searchsorted(E2M1_EDGES, q).astype(np.int64)
    decoded = np.where(groups < 0, -1.0, 1.0) * E2M1_VALUES[codes] * vq
    return decoded.reshape(x.shape)


def quantize_int8_group64(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    tail = x.shape[-1] % 64
    if tail:
        pad = 64 - tail
        x = np.concatenate([x, np.zeros((*x.shape[:-1], pad), dtype=np.float64)], axis=-1)
    groups = x.reshape(*x.shape[:-1], -1, 64)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 127.0, 1e-30)
    scale = scale.astype(np.float16).astype(np.float64)
    q = np.clip(np.round(groups / scale), -127, 127)
    decoded = q * scale
    if tail:
        decoded = decoded[..., :tail]
    return decoded.reshape(x.shape)


def e4m3fn_data_code(x: np.ndarray) -> np.ndarray:
    """E4M3FN DATA-plane BYTES -- the kernel's gqa_kv_nvfp4_data_fp32_to_e4m3.

    Lifted clause by clause from the post-image of
    FP8E8-FIX2/patch/0001-fp8-e4m3-data-pair.patch@5e5c9584bfd6ad79
    (src/ops/kernel/gqa_attention_kv_nvfp4.cuh), DATA lattice only:
        x != x          -> 0x00      a NaN must not become a large FINITE value
        exponent >= 16  -> 0x7E      "|x| >= 512: saturate to the format max, 448"
        exponent <= 0   -> rintf(ax * 512), ties-to-EVEN, and 0x08 above 2^-6
        otherwise       -> round-to-nearest-even on the 3-bit mantissa
        code >= 0x7F    -> 0x7E      "0x7F is a NaN code: clamp to 448"
    The SCALE lattice is a different kernel function (gqa_kv_nvfp4_e4m3_magnitude_bits
    behind gqa_kv_nvfp4_fp32_to_e4m3) and is deliberately NOT reproduced here: that
    one freezes its top octave at 0x7F, and 480.0 is the right reading of it.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    sign = ((bits >> np.uint32(24)) & np.uint32(0x80)).astype(np.uint8)
    a = np.abs(x)
    ab = np.ascontiguousarray(a).view(np.uint32)
    exp = ((ab >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64) - 127 + 7
    mag = np.zeros(ab.shape, dtype=np.uint8)
    hi = exp >= 16
    mag[hi] = np.uint8(0x7E)
    lo = exp <= 0
    if lo.any():
        v = np.rint(a[lo].astype(np.float64) * 512.0).astype(np.int64)
        np.clip(v, 0, None, out=v)
        r = v.astype(np.uint8)
        r[v >= 8] = np.uint8(0x08)
        mag[lo] = r
    mid = (~hi) & (~lo)
    if mid.any():
        bm = ab[mid]
        m = ((bm >> np.uint32(20)) & np.uint32(0x7)).astype(np.int64)
        g = ((bm >> np.uint32(19)) & np.uint32(1)).astype(np.int64)
        s = (bm & np.uint32(0x7FFFF)).astype(np.int64)
        e = exp[mid]
        carry = (g == 1) & ((s != 0) | ((m & 1) == 1))
        m2 = np.where(carry, m + 1, m)
        wrap = carry & (m2 > 7)
        e2 = np.where(wrap, e + 1, e)
        code = (e2 << 3) | np.where(wrap, 0, m2)
        code = np.where(e2 >= 16, 0x7E, code)
        code = np.where(code >= 0x7F, 0x7E, code)
        mag[mid] = code.astype(np.uint8)
    return np.where(nan, np.uint8(0), (sign | mag).astype(np.uint8))


def e4m3fn_data_code_to_f32(code: np.ndarray) -> np.ndarray:
    """E4M3FN DATA-plane read -- the kernel's gqa_kv_nvfp4_data_e4m3_to_f32.

    `mag = byte & 0x7F`, so bit 7 IS the sign here (on the scale plane it is not).
    The NaN codes 0x7F / 0xFF read back as +-448.0 -- the format maximum -- and
    NOT as 480.0.
    """
    code = np.asarray(code, dtype=np.int64)
    mag = code & 0x7F
    e = (mag >> 3) & 0x0F
    m = mag & 0x07
    magnitude = np.where(e == 0, m.astype(np.float64) / 512.0,
                         np.ldexp(1.0 + m.astype(np.float64) / 8.0, e - 7))
    magnitude = np.where((e == 15) & (m == 7), 448.0, magnitude)
    return np.where((code & 0x80) != 0, -magnitude, magnitude)

def quantize_fp8_group16(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 448.0, 2.0**-9)
    q = np.clip(np.round(groups / scale), -448, 448)
    decoded = q * scale
    return decoded.reshape(x.shape)


# The uniform-grid body above is kept BYTE FOR BYTE as the pre-image reference, so
# both lattices can be measured in one process; every caller in this file resolves
# the E4M3FN one, because the name is rebound below. Nothing above was rewritten:
# this patch is insertion-only, so no CRLF-carrying line was regenerated.
quantize_fp8_group16_uniform_grid = quantize_fp8_group16


def quantize_fp8_group16(x: np.ndarray) -> np.ndarray:
    """fp8 group-16 through the E4M3FN DATA codebook, matching the post-B kernel.

    The DIVISOR IS NOT TOUCHED. The kernel writes
        kscale = fmaxf(kmax / 448.0f, 0.001953125f)
    and the writer divides by that raw float -- `scale` below is still exactly the
    expression it was. Only the lattice the quotient lands on changes: it used to
    be a uniform integer grid clipped to +-448 (~9.8 bits of resolution, which made
    the fp8 column 1003x..1010x too optimistic on the 176-frame corpus), and it is
    now the E4M3FN byte lattice that gqa_kv_nvfp4_data_fp32_to_e4m3 /
    _data_e4m3_to_f32 imply:
      * 3 mantissa bits, so 8 levels per octave and up to 12.5% relative step;
      * the FULL top octave, [248, 448] saturating at 0x7E = 448, instead of the
        writer collapsing everything at or above 248 onto the NaN code 0x7F, which
        its reader then returns as 480.0 = 1.07x the group's own amax. Measured on
        the 176-frame corpus, that collapse covered 98.32% of all elements and
        100.00% of the groups, so every group's amax was biased by +7.1429%;
      * ties-to-even in the denormal branch (rintf), not the scale path's roundf.
    """
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 448.0, 2.0**-9)
    code = e4m3fn_data_code(np.ascontiguousarray((groups / scale).astype(np.float32)))
    decoded = e4m3fn_data_code_to_f32(code) * scale
    return decoded.reshape(x.shape)

def quantize_iso4e_group16(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 7.0, 2.0**-9)
    q = np.clip(np.round(groups / scale), -7, 7)
    decoded = q * scale
    return decoded.reshape(x.shape)


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    num = np.sum((x - y) ** 2)
    den = np.sum(x**2)
    return float(num / den) if den > 0 else 0.0


def effective_rank(x: np.ndarray) -> float:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] > x.shape[1]:
        eig = np.linalg.eigvalsh(x.T @ x)
    else:
        eig = np.linalg.eigvalsh(x @ x.T)
    s = np.sqrt(np.maximum(eig, 0.0))
    if s.size == 0 or s[0] <= 0:
        return 0.0
    p = s / s[0]
    den = np.sum(p**2)
    return float(np.sum(p) ** 2 / den) if den > 0 else 0.0


def outlier_scores(x: np.ndarray) -> tuple[float, float]:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    rms = math.sqrt(float(np.mean(x**2)))
    if rms == 0:
        return 0.0, 0.0
    head = float(np.mean((np.abs(x).max(axis=-1) > 6.0 * rms).astype(np.float64)))
    dim_group = float(
        np.mean(
            (
                np.abs(x).max(axis=-2)
                > 6.0 * rms * math.sqrt(x.shape[-2])
            ).astype(np.float64)
        )
    )
    return head, dim_group


def load_frames(directory: Path):
    frames = []
    for path in sorted(directory.glob("*.kvc")):
        raw = path.read_bytes()
        if len(raw) < HEADER.size:
            raise SystemExit(f"truncated record: {path}")
        magic, header_bytes, layer, head_dim, kv_heads, tokens, record_index, first_pos, last_pos, *_ = (
            HEADER.unpack(raw[: HEADER.size])
        )
        if magic != MAGIC or header_bytes != HEADER.size:
            raise SystemExit(f"bad record header: {path}")
        payload = np.frombuffer(raw, dtype=np.uint8, offset=HEADER.size)
        expect = (tokens * 4) + 2 * head_dim * kv_heads * tokens * 2
        if payload.size != expect:
            raise SystemExit(f"bad record payload: {path}")
        pos = payload[: tokens * 4].copy().view(np.int32)
        arr = np.frombuffer(payload[tokens * 4 :].tobytes(), dtype="<u2").view(np.float16)
        half = head_dim * kv_heads * tokens
        k = arr[:half].reshape((head_dim, kv_heads, tokens), order="F").transpose(2, 1, 0)
        v = arr[half:].reshape((head_dim, kv_heads, tokens), order="F").transpose(2, 1, 0)
        k = k.astype(np.float32)
        v = v.astype(np.float32)
        frames.append(
            {
                "path": str(path),
                "layer": layer,
                "record": record_index,
                "tokens": tokens,
                "positions": pos.astype(np.int32),
                "k": k,
                "v": v,
            }
        )
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="directory of .kvc frames")
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--budget", type=float, default=1.20,
                    help="memory budget relative to all-nvfp4 (default 1.20)")
    ap.add_argument("--k-weight", type=float, default=0.65,
                    help="MixKVQ K error weight; V gets 1-k-weight")
    args = ap.parse_args()

    frames = load_frames(Path(args.records))
    if not frames:
        raise SystemExit("no .kvc frames found")
    layers = sorted({f["layer"] for f in frames})
    if layers != list(range(len(layers))):
        raise SystemExit("records are not a contiguous 0..N-1 full-attention layer set")
    n_layers = len(layers)
    if n_layers > 16:
        raise SystemExit(f"too many layers for the runtime table: {n_layers}")

    # Layer statistics accumulated over frames with token weighting.
    stats = {
        layer: {
            "tokens": 0,
            "k_nmse": {"nvfp4": 0.0, "int8": 0.0, "fp8": 0.0, "iso4e": 0.0},
            "v_nmse": {"nvfp4": 0.0, "int8": 0.0, "fp8": 0.0, "iso4e": 0.0},
            "k_rank": 0.0,
            "v_rank": 0.0,
            "k_outlier_head": 0.0,
            "v_outlier_head": 0.0,
            "k_outlier_group": 0.0,
            "v_outlier_group": 0.0,
        }
        for layer in layers
    }

    for frame in frames:
        layer = frame["layer"]
        weight = frame["tokens"]
        stat = stats[layer]
        stat["tokens"] += weight
        k = frame["k"]
        v = frame["v"]
        stat["k_nmse"]["nvfp4"] += nmse(k, quantize_e2m1_group16(k)) * weight
        stat["k_nmse"]["int8"] += nmse(k, quantize_int8_group64(k)) * weight
        stat["k_nmse"]["fp8"] += nmse(k, quantize_fp8_group16(k)) * weight
        stat["k_nmse"]["iso4e"] += nmse(k, quantize_iso4e_group16(k)) * weight
        stat["v_nmse"]["nvfp4"] += nmse(v, quantize_e2m1_group16(v)) * weight
        stat["v_nmse"]["int8"] += nmse(v, quantize_int8_group64(v)) * weight
        stat["v_nmse"]["fp8"] += nmse(v, quantize_fp8_group16(v)) * weight
        stat["v_nmse"]["iso4e"] += nmse(v, quantize_iso4e_group16(v)) * weight
        for h in range(k.shape[1]):
            stat["k_rank"] += effective_rank(k[:, h, :]) * weight
            stat["v_rank"] += effective_rank(v[:, h, :]) * weight
        ok_head, ok_group = outlier_scores(k)
        ov_head, ov_group = outlier_scores(v)
        stat["k_outlier_head"] += ok_head * weight
        stat["v_outlier_head"] += ov_head * weight
        stat["k_outlier_group"] += ok_group * weight
        stat["v_outlier_group"] += ov_group * weight

    rows = []
    for layer in layers:
        stat = stats[layer]
        w = stat["tokens"]
        k_err = stat["k_nmse"]["nvfp4"] / w
        v_err = stat["v_nmse"]["nvfp4"] / w
        mix_err = args.k_weight * k_err + (1.0 - args.k_weight) * v_err
        triaxial = max(
            stat["k_outlier_head"], stat["v_outlier_head"],
            stat["k_outlier_group"], stat["v_outlier_group"],
        ) / max(w, 1)
        rank = 0.5 * (stat["k_rank"] / w + stat["v_rank"] / w)
        # Normalized 0..1 scores across layers for greedy ranking.
        rows.append(
            {
                "layer": layer,
                "tokens": w,
                "nmse_nvfp4_k": k_err,
                "nmse_nvfp4_v": v_err,
                "nmse_int8_k": stat["k_nmse"]["int8"] / w,
                "nmse_int8_v": stat["v_nmse"]["int8"] / w,
                "nmse_fp8_k": stat["k_nmse"]["fp8"] / w,
                "nmse_fp8_v": stat["v_nmse"]["fp8"] / w,
                "nmse_iso4e_k": stat["k_nmse"]["iso4e"] / w,
                "nmse_iso4e_v": stat["v_nmse"]["iso4e"] / w,
                "mixkvq_error": mix_err,
                "triaxial_outlier": triaxial,
                "arkv_rank": rank,
            }
        )
    for key in ("mixkvq_error", "triaxial_outlier", "arkv_rank"):
        lo = min(row[key] for row in rows)
        hi = max(row[key] for row in rows)
        for row in rows:
            row[f"{key}_norm"] = 0.0 if hi <= lo else (row[key] - lo) / (hi - lo)
    for row in rows:
        row["sensitivity"] = (
            0.50 * row["mixkvq_error_norm"]
            + 0.25 * row["triaxial_outlier_norm"]
            + 0.25 * row["arkv_rank_norm"]
        )

    # Storage cost per token per layer (K+V), in bytes; nvfp4 is the budget base.
    head_dim = frames[0]["k"].shape[-1]
    kv_heads = frames[0]["k"].shape[1]
    def cost(dtype: str) -> float:
        if dtype == "nvfp4" or dtype == "iso4e":
            return kv_heads * (head_dim + 2 * (head_dim / 16))
        if dtype == "int8":
            return kv_heads * (2 * head_dim + 4 * (head_dim / 64))
        if dtype == "fp8":
            return kv_heads * (2 * head_dim + 2 * (head_dim / 16))
        return kv_heads * 4 * head_dim

    base = n_layers * cost("nvfp4")
    ranked = sorted(rows, key=lambda row: row["sensitivity"], reverse=True)
    table = {layer: "nvfp4" for layer in layers}
    used = base
    for row in ranked:
        for dtype in ("fp8", "int8", "bf16"):
            delta = cost(dtype) - cost(table[row["layer"]])
            if used + delta <= args.budget * base:
                table[row["layer"]] = dtype
                used += delta
                break

    spec = ",".join(f"{layer}:{table[layer]}" for layer in layers)
    report = {
        "record_frames": len(frames),
        "layer_count": n_layers,
        "head_dim": head_dim,
        "kv_heads": kv_heads,
        "budget_factor": args.budget,
        "k_weight": args.k_weight,
        "per_layer": sorted(rows, key=lambda row: row["layer"]),
        "selected_table": table,
        "kv_layer_storage_spec": spec,
        "relative_cost": used / base,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"frames={len(frames)} layers={n_layers} budget={args.budget:.2f}x -> "
          f"cost={used / base:.3f}x")
    print("--kv-layer-storage " + spec)


if __name__ == "__main__":
    main()
