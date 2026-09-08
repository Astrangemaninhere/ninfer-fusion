# -*- coding: utf-8 -*-
"""ft_tiers.py — FreeToken 步 2 决策逻辑: 从 ft_stats 观测数据生成逐层 KV 方案.

输入: serve stderr 的 '[ft] layer=N mean_l=X rounds=M' 行 (NINFER_FT_STATS=1 产出)
策略: 深层保护 (§46/§47 验收) + 能量分位 (§41 步 2):
  - 深层 (最后 20% 全注意力层) 强制 NVFP4 — 深层敏感, 不参与压缩
  - 其余层按 mean_l 分位切 3 挡: 高能量=iso3, 中=nvfp4, 低=e8
输出: --kv-layer-storage '...' 规格串 (可直接透传 serve)

用法: python3 ft_tiers.py <serve_log> [层总数] [--deep-frac 0.2]
"""
from __future__ import annotations

import re
import sys

LINE_RE = re.compile(r"\[ft\] layer=(\d+) mean_l=([\d.eE+-]+) rounds=(\d+)")


def parse_stats(log: str) -> dict[int, float]:
    # 取每层最后一次 (rounds 最大) 的 mean_l
    best: dict[int, tuple[int, float]] = {}
    for m in LINE_RE.finditer(log):
        layer, mean_l, rounds = int(m[1]), float(m[2]), int(m[3])
        if layer not in best or rounds >= best[layer][0]:
            best[layer] = (rounds, mean_l)
    return {k: v for k, (_, v) in sorted(best.items())}


def build_spec(energy: dict[int, float], total_full_attn: int,
               deep_frac: float = 0.2,
               tiers: tuple[str, str, str] = ("e8", "iso3", "nvfp4")) -> str:
    deep_start = int(total_full_attn * (1 - deep_frac))
    deep = [l for l in range(total_full_attn) if l >= deep_start]
    measured = {l: e for l, e in energy.items() if l < deep_start}
    # 无观测数据的层保守给 nvfp4
    unmeasured = [l for l in range(deep_start) if l not in measured]

    low, mid, hi = tiers  # 能量升序: 低能量层最耐压
    parts = [f"{l}:{hi}" for l in deep]                    # 深层保 NVFP4
    parts += [f"{l}:{mid}" for l in unmeasured]
    if measured:
        items = sorted(measured.items(), key=lambda kv: kv[1])  # 能量升序
        n = len(items)
        n_low = max(1, n // 3)
        for i, (l, _e) in enumerate(items):
            # 能量最低的 1/3 -> e8, 中间 1/3 -> iso3, 高能量 -> nvfp4
            t = low if i < n // 3 else (mid if i < 2 * (n // 3) else hi)
            parts.append(f"{l}:{t}")
    return ",".join(parts)


def main() -> int:
    log = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1], encoding='utf-8', errors='replace').read()
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    energy = parse_stats(log)
    print(f"观测层数: {len(energy)} / {total}")
    for l in sorted(energy):
        print(f"  layer {l:2d}: mean_l={energy[l]:.4g}")
    spec = build_spec(energy, total)
    print("== 生成方案 ==")
    print(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
