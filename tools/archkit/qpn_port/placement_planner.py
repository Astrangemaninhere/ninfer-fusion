#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""placement_planner.py — 特性驱动的异构任务分配器原型 (TODO §0-A, L1/L2).

核心原则 (用户定调): 按各部分的特性分配任务性质, 使整体效率最大化。

负载特性模型 (decode 与 prefill 分开建模):
  decode  (batch=1, 带宽型): t = (weight_bytes + kv_bytes(ctx)) / bw
  prefill (compute 型):     t = flops / compute + kv_bytes(ctx) / bw
  草稿/查表 (延迟型):        t ≈ launch_us + work
设备画像由 device_profile.cu 实测输出 (compute_tflops / bw_gbs / launch_us /
int8_gops), 同一内核跨设备测量 -> 数值可直接横向比较。

用法:
  python3 placement_planner.py --demo          # 内置 5090+V100+CPU 演示
  python3 placement_planner.py --profile dev.json --model qwen27
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field


@dataclass
class Device:
    name: str
    kind: str                      # gpu-new / gpu-old / cpu / igpu / npu
    compute_tflops: float          # 同内核实测标量算力 (横向可比)
    bw_gbs: float                  # 实测有效带宽
    launch_us: float               # 内核启动延迟
    int8_gops: float = 0.0
    vram_gb: float = 0.0           # 权重驻留上限

    def decode_time_s(self, weight_bytes: float, kv_bytes: float) -> float:
        return (weight_bytes + kv_bytes) / (self.bw_gbs * 1e9)

    def prefill_time_s(self, flops: float, kv_bytes: float) -> float:
        return flops / (self.compute_tflops * 1e12) + kv_bytes / (self.bw_gbs * 1e9)


@dataclass
class Layer:
    idx: int
    weight_bytes: float            # 本层权重字节数 (按目标精度)
    flops_per_token: float
    kv_bytes_per_token: float      # K+V 每 token 字节

    def kv_bytes(self, ctx: int) -> float:
        return self.kv_bytes_per_token * ctx


def qwen27_layers(n_layers: int = 64, hidden: int = 5120, inter: int = 17408,
                  kv_heads: int = 4, head_dim: int = 256, wbits: float = 4.0) -> list[Layer]:
    """按模型几何生成逐层负载画像 (NVFP4 ≈ 0.5625 B/权重 -> wbits≈4.5, 用 4.0 保守)."""
    per_w = wbits / 8.0
    attn_w = 4 * hidden * hidden * per_w            # q/k/v/o
    mlp_w = 3 * hidden * inter * per_w              # gate/up/down
    kv = 2 * kv_heads * head_dim * 2.0              # K+V bf16 字节 / token
    out = []
    for i in range(n_layers):
        out.append(Layer(i, attn_w + mlp_w, 2 * (attn_w + mlp_w), kv))
    return out


def plan_placement_capacity(devices: list[Device], layers: list[Layer], ctx: int,
                            regime: str, caps: dict[str, float]) -> dict:
    """容量受限版: 权重字节数不得超过设备 vram; 单卡装不下 => 强制多卡切分.
    保持层连续 (每设备一段), 穷举连续切分点取总时间最小."""
    import itertools
    n = len(layers)
    best = None
    # 连续段划分: devices 数 k, 穷举 k-1 个切点 (层有序), 每段检查容量
    for cuts in itertools.combinations(range(1, n), len(devices) - 1):
        bounds = (0,) + cuts + (n,)
        segs = []
        ok = True
        for di, (a, b) in enumerate(zip(bounds, bounds[1:])):
            wbytes = sum(L.weight_bytes for L in layers[a:b])
            if wbytes > caps.get(devices[di].name, 0) * 1e9:
                ok = False
                break
            segs.append((di, a, b))
        if not ok:
            continue
        total = 0.0
        per = {}
        for di, a, b in segs:
            d = devices[di]
            t = 0.0
            for L in layers[a:b]:
                t += (d.decode_time_s(L.weight_bytes, L.kv_bytes(ctx)) if regime == "decode"
                      else d.prefill_time_s(L.flops_per_token, L.kv_bytes(ctx)))
            total += t
            per[d.name] = {"layers": f"{a}-{b - 1}", "count": b - a,
                           "est_ms_per_token": round(t * 1e3, 3)}
        # 串行流水: 每 token 依次过各段 -> 总时间 = 各段之和
        if best is None or total < best["est_ms_per_token_s"]:
            best = {"per_device": per, "est_ms_per_token_s": total,
                    "cut_at_layers": list(bounds[1:-1])}
    if best is None:
        return {"error": "容量约束下无可行划分 (权重总量超所有卡之和)"}
    solo = min(
        (sum((d.decode_time_s(L.weight_bytes, L.kv_bytes(ctx)) if regime == "decode"
              else d.prefill_time_s(L.flops_per_token, L.kv_bytes(ctx)))
             for L in layers), d.name)
        for d in devices if d.vram_gb * 1e9 >= sum(L.weight_bytes for L in layers)
    ) if any(d.vram_gb * 1e9 >= sum(L.weight_bytes for L in layers) for d in devices) \
        else (float("inf"), "none (单卡装不下)")
    best["best_single_device"] = {"name": solo[1], "ms": round(solo[0] * 1e3, 3)}
    best["speedup_vs_best_single"] = round(solo[0] / best["est_ms_per_token_s"], 2) \
        if solo[0] != float("inf") else "single-device infeasible"
    best["capacity_note"] = "bf16 47.6GB 超 5090 32GB => 强制切分到 V100"
    best["regime"], best["ctx"] = regime, ctx
    return best


def plan_placement(devices: list[Device], layers: list[Layer], ctx: int,
                   regime: str = "decode", hysteresis: float = 0.15) -> dict:
    """贪心 argmin + 迟滞: 每层放到该 regime 下最快的设备; 设备切换需要
    新设备快过当前设备 hysteresis 比例才切换 (避免乒乓)。"""
    times: list[list[float]] = []
    for L in layers:
        kv = L.kv_bytes(ctx)
        row = [d.decode_time_s(L.weight_bytes, kv) if regime == "decode"
               else d.prefill_time_s(L.flops_per_token, kv) for d in devices]
        times.append(row)

    assign: list[int] = []
    cur = min(range(len(devices)), key=lambda d: times[0][d])
    for row in times:
        best = min(range(len(devices)), key=lambda d: row[d])
        if best != cur and row[best] * (1 + hysteresis) < row[cur]:
            cur = best
        assign.append(cur)

    total = sum(times[i][assign[i]] for i in range(len(layers)))
    per_dev: dict[str, dict] = {}
    for di, d in enumerate(devices):
        idxs = [i for i, a in enumerate(assign) if a == di]
        t = sum(times[i][di] for i in idxs)
        if idxs:
            per_dev[d.name] = {
                "kind": d.kind, "layers": f"{idxs[0]}-{idxs[-1]}",
                "count": len(idxs), "est_ms_per_token": round(t * 1e3, 3),
            }
    solo = min(
        (sum(times[i][di] for i in range(len(layers))), devices[di].name)
        for di in range(len(devices))
    )
    return {
        "regime": regime, "ctx": ctx,
        "assignment": assign,
        "per_device": per_dev,
        "est_ms_per_token": round(total * 1e3, 3),
        "best_single_device": {"name": solo[1], "ms": round(solo[0] * 1e3, 3)},
        "speedup_vs_best_single": round(solo[0] / total, 2),
    }


def demo() -> None:
    # 设备画像: 5090 为 device_profile.cu 实测; V100/CPU 为规格推算 (待实测替换)
    dev_5090 = Device("RTX5090D", "gpu-new", 1.3, 1625, 8.0, 456, 32)
    dev_v100 = Device("V100-32G", "gpu-old", 0.9, 900, 6.0, 120, 32)
    dev_cpu = Device("CPU", "cpu", 0.05, 50, 100.0, 30, 0)

    layers = qwen27_layers()
    for regime, ctx in (("decode", 2048), ("prefill", 2048)):
        plan = plan_placement([dev_5090, dev_v100, dev_cpu], layers, ctx, regime)
        print(f"== {regime} @ ctx={ctx}")
        print(json.dumps({k: plan[k] for k in ("per_device", "est_ms_per_token",
                                               "best_single_device",
                                               "speedup_vs_best_single")},
                         ensure_ascii=False, indent=1))
    # 场景 2 (容量受限, L1 真实用例): bf16 权重 47.6GB > 5090 32GB 显存
    # => 溢出层被迫放 V100, 规划器需在容量约束下求最优切分
    layers_bf16 = qwen27_layers(wbits=16.0)
    for regime, ctx in (("decode", 2048),):
        plan = plan_placement_capacity(
            [dev_5090, dev_v100], layers_bf16, ctx, regime,
            caps={"RTX5090D": dev_5090.vram_gb, "V100-32G": dev_v100.vram_gb})
        print(f"== {regime} @ ctx={ctx} [bf16 容量受限]")
        if "error" in plan:
            print(json.dumps(plan, ensure_ascii=False, indent=1))
            continue
        print(json.dumps({k: plan[k] for k in ("per_device", "est_ms_per_token_s",
                                               "cut_at_layers", "best_single_device",
                                               "speedup_vs_best_single",
                                               "capacity_note")}
                         if "error" not in plan else plan,
                         ensure_ascii=False, indent=1))
    # 草稿/查表 (延迟型) 演示: 单次小核调用
    print("== draft/lookup (延迟型)")
    for d in (dev_5090, dev_v100, dev_cpu):
        print(f"  {d.name}: launch {d.launch_us}us -> 延迟型负载最优: "
              f"{'CPU/igpu' if d.launch_us > 50 else 'GPU'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
