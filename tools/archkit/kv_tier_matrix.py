#!/usr/bin/env python3
"""kv_tier_matrix.py — 自动化 KV 档位 × 层 标定流水线。

目标：为「哪一层用哪种 KV 量化」提供**可量化比较**的依据，判据同时含精度与速度，
且全部由实测产生、可自动复跑（不是一次性实验）。

测什么（全档位，自动展开）：
  A) 统一档位臂：16 层同档 ∈ {bf16, int8, fp8, nvfp4, iso3, e8}
     → 隔离「每档反量化/读取成本」。注意实际路径（勿凭档名推断）：
        - int8 活跃内核**不做旋转**（单趟 mma_s8 + 每 64 元素一个 fp16 scale）；
        - nvfp4 = E2M1 原生 QK（但**无条件跑第二趟残差 QK**）+ SO(4) 旋转 + Sinkhorn row scale
          + V 走软件解码(ISO3) + BF16 PV；
        - iso3 是独立内核：Iso3Group16 映射到 DType::ISO3，走
          gqa_attention_decode_iso3.cuh（K 面是 ISO3 符号幅值 nibble，**不是 E2M1**）；
          与 nvfp4 只共享平面几何（两 code/byte + per-16 E4M3FN scale ⇒ 4.50 bits/el）、
          不共享内核，也不读 nvfp4 的第二级残差平面（decoder_state.cpp:174-196）。
          （旧注释写「映射到 DType::NVFP4 / 同内核」，已过时。）
        - e8 = int8 内核 + E8 格点投影/H64 + 每 64 元素 scale。
  B) 单层差分臂：基线档 nvfp4，仅把第 L 层换成 e8（L = 0..15）
     → 每层对「换便宜档」的**速度收益**与**精度代价**的归因（可加性稍后用双层臂校验）
  C) 可加性校验：同时换两层（首/中/尾各一组）
  D) 默认臂：出厂混合表（10×e8 + 6×nvfp4）作为对照

每个臂的读数：decode tok/s、AL、graded 答案（长上下文针尖精确前缀长度 0..N）、
KV payload、审计行里解析出的档位混合。

输出：JSON（机器可读，供分配器/DP 消费）+ 排序表 + 首版建议。

用法（在本机，长跑务必后台）：
  python3 tools/archkit/kv_tier_matrix.py --out /path/table.json [--quick]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

MODEL = os.environ.get("KVX_MODEL", "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer")
BIN = os.environ.get("KVX_BIN", "/home/user/ninfer-fusion/build/apps/ninfer")
MSGS = os.environ.get(
    "KVX_MESSAGES",
    "/home/user/ninfer-fusion/examples/cli/messages/long_niah_64k.json")
NEEDLE = os.environ.get("KVX_NEEDLE", "ORCHID=493817; COLOR=COBALT")
TIERS = ["bf16", "int8", "fp8", "nvfp4", "iso3", "e8"]
LAYERS = 16


def spec_with(base, overrides):
    """base 为全局默认档，overrides = {layer: tier}；输出不重叠的连续区间语法。"""
    tiers = {}
    for l in range(LAYERS):
        tiers[l] = overrides.get(l, base)
    parts, start = [], 0
    for l in range(1, LAYERS + 1):
        if l == LAYERS or tiers[l] != tiers[start]:
            lo, hi = start, l - 1
            parts.append(f"{lo}:{tiers[start]}" if lo == hi else f"{lo}-{hi}:{tiers[start]}")
            start = l
    return ",".join(parts)

def run_arm(label, extra, max_new=32, timeout=1800):
    """跑一个臂，返回读数 dict。长上下文 + 贪心 ⇒ 确定性，单次即可。"""
    subprocess.run(["pkill", "-9", "-x", "ninfer"], check=False)
    time.sleep(3)
    argv = [BIN, MODEL, "--messages", MSGS, "--max-new", str(max_new),
            "--max-context", "131072", "--kv-capacity", "auto", "--no-thinking",
            "--greedy", "--spec", "mtp", "--draft-tokens", "9"] + list(extra)
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "timeout"
    wall = time.time() - t0

    def metric(pattern, group=1):
        m = re.search(pattern, err)
        return m.group(group) if m else None

    # graded：长上下文针尖答案的精确前缀长度（0..len(NEEDLE)）
    flat = re.sub(r"\s+", " ", out)
    graded = 0
    while graded < len(NEEDLE) and graded < len(flat) and flat[graded] == NEEDLE[graded]:
        graded += 1
    mix = metric(r"\[kv-bit-budget\][^\n]*") if "--kv-bit-budget" in " ".join(extra) else None
    return {
        "label": label, "extra": list(extra), "rc": rc, "wall_s": round(wall, 1),
        "decode_tok_s": metric(r"decode speed\s+([\d.]+)"),
        "prefill_tok_s": metric(r"prefill speed\s+([\d.]+)"),
        "al": metric(r"acceptance length\s+([\d.]+)"),
        "kv_payload": metric(r"kv cache payload\s+([\d.]+ \w+)"),
        "kv_capacity": metric(r"KV capacity\s+(\d+)"),
        "graded_hit": graded, "graded_max": len(NEEDLE),
        "needle_ok": graded == len(NEEDLE),
        "hot": (mix or "").split("hot=")[-1].split(" (")[0] if mix else None,
        "error": metric(r"^error: (.+)$"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="只跑统一档位臂")
    args = ap.parse_args()

    arms = []
    # A) 统一档位
    arms.append(("uniform_bf16", ["--kv-dtype", "bf16"]))
    for t in TIERS[1:]:
        arms.append((f"uniform_{t}", ["--kv-layer-storage", f"0-{LAYERS-1}:{t}"]))
    # D) 默认
    arms.append(("default_mixed", []))
    if not args.quick:
        # B) 单层差分（基线 nvfp4，逐层换 e8）
        for l in range(LAYERS):
            arms.append((f"only_layer{l}_e8",
                         ["--kv-layer-storage", spec_with("nvfp4", {l: "e8"})]))
        # C) 可加性：两层同时换
        for pair in ((0, 1), (7, 8), (14, 15)):
            arms.append((f"pair_{pair[0]}_{pair[1]}_e8",
                         ["--kv-layer-storage", spec_with("nvfp4", {pair[0]: "e8", pair[1]: "e8"})]))

    results = []
    for label, extra in arms:
        r = run_arm(label, extra)
        results.append(r)
        print(f"[{label}] rc={r['rc']} tok/s={r['decode_tok_s']} "
              f"hit={r['graded_hit']}/{r['graded_max']} payload={r['kv_payload']} "
              f"hot={r['hot']}", flush=True)

    with open(args.out, "w") as fh:
        json.dump({"model": MODEL, "needle": NEEDLE, "arms": results}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
