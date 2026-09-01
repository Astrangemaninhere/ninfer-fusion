#!/usr/bin/env python
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
        return 448.0 / 512.0 * 2**7  # saturate (0b01111111 = 448)
    if exp <= 0:
        mant = int(round(x * 64.0))
        if mant <= 0:
            return 0.0
        if mant >= 8:
            return 1.0
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
                return 448.0 / 512.0 * 2**7
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


def quantize_fp8_group16(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    groups = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 448.0, 2.0**-9)
    q = np.clip(np.round(groups / scale), -448, 448)
    decoded = q * scale
    return decoded.reshape(x.shape)


def quantize_iso3_group16(x: np.ndarray) -> np.ndarray:
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
            "k_nmse": {"nvfp4": 0.0, "int8": 0.0, "fp8": 0.0, "iso3": 0.0},
            "v_nmse": {"nvfp4": 0.0, "int8": 0.0, "fp8": 0.0, "iso3": 0.0},
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
        stat["k_nmse"]["iso3"] += nmse(k, quantize_iso3_group16(k)) * weight
        stat["v_nmse"]["nvfp4"] += nmse(v, quantize_e2m1_group16(v)) * weight
        stat["v_nmse"]["int8"] += nmse(v, quantize_int8_group64(v)) * weight
        stat["v_nmse"]["fp8"] += nmse(v, quantize_fp8_group16(v)) * weight
        stat["v_nmse"]["iso3"] += nmse(v, quantize_iso3_group16(v)) * weight
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
                "nmse_iso3_k": stat["k_nmse"]["iso3"] / w,
                "nmse_iso3_v": stat["v_nmse"]["iso3"] / w,
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
        if dtype == "nvfp4" or dtype == "iso3":
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
