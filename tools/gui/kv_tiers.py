#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kv_tiers.py — KV 分层精度自由组合的契约 (2026-09-03 定调).

三层 (热/尾/冷) x 多种 KV 量化格式, 组合不固化:
  --kv-tier-formats hot=bf16,tail=fp16,cold=iso3
每层可独立选: bf16 / fp16 / int8 / int4 / iso4 / iso3 / e8(2bit) / auto

- auto = 该层用引擎默认 (热=权重档 KV dtype, 尾=同热, 冷=iso3)
- 尾层与冷层窗口大小仍走 --kv-tail-tokens / --cold-keep-tokens (另文契约)
- 本模块只做语法/取值校验, 返回规范化 dict; C++ 引擎侧同语法待实现
"""
from __future__ import annotations

import re

TIERS = ("hot", "tail", "cold")
FORMATS = ("bf16", "fp16", "int8", "int4", "iso4", "iso3", "e8", "auto")
# nvfp4 模式 (权重档拆分, 2026-09-03 定调):
#   fusion = 现融合版全部特性 (E8/冷池/尾窗等全格式可用)
#   pure   = 纯 nvfp4 基线 (对照归因): 只允许经典格式, 禁 e8/iso 系与冷编码
MODES = ("fusion", "pure")
MODE_FORMATS = {
    "fusion": set(FORMATS),
    "pure": {"bf16", "fp16", "int8", "int4", "auto"},
}
# 语义分组 (注释/提示用): 比特数约值
FORMAT_BITS = {"bf16": 16, "fp16": 16, "int8": 8, "int4": 4,
               "iso4": 4, "iso3": 3, "e8": 2, "auto": None}
DEFAULTS = {"hot": "auto", "tail": "auto", "cold": "iso3"}
PURE_DEFAULTS = {"hot": "auto", "tail": "auto", "cold": "int8"}   # pure: 经典档默认

_ITEM = re.compile(r"^(hot|tail|cold)=(bf16|fp16|int8|int4|iso4|iso3|e8|auto)$")


def parse(spec: str | None, mode: str = "fusion") -> dict:
    """解析 --kv-tier-formats。mode=fusion|pure 限制可用格式集。
    返回 {hot:.., tail:.., cold:..}; 非法输入抛 ValueError。"""
    if mode not in MODES:
        raise ValueError("bad mode %r (want %s)" % (mode, "/".join(MODES)))
    out = dict(PURE_DEFAULTS if mode == "pure" else DEFAULTS)
    if not spec or not spec.strip():
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        m = _ITEM.match(item)
        if not m:
            raise ValueError(
                "bad --kv-tier-formats item %r (want tier=format, e.g. cold=iso3; "
                "tiers=%s formats=%s)" % (item, "/".join(TIERS), "/".join(FORMATS)))
        tier, fmt = m.group(1), m.group(2)
        if mode == "pure" and fmt not in MODE_FORMATS["pure"]:
            raise ValueError("nvfp4-mode=pure forbids %s (fusion-only format); "
                             "pure = 经典格式基线对照" % fmt)
        # 热层不允许降精度到 < int8 (活动页是每次解码都读的, 有意为之)
        if tier == "hot" and fmt in ("int4", "iso4", "iso3", "e8"):
            raise ValueError("hot tier cannot use %s (activity pages decode-hot); "
                             "use bf16/fp16/int8 or auto" % fmt)
        # 尾层意义 = 高于或等于热层 (精度尾), 低于热层没有意义
        hot_bits = FORMAT_BITS.get(out["hot"]) or 16   # auto 按 16 位默认算
        if tier == "tail" and fmt != "auto" and (FORMAT_BITS.get(fmt) or 16) < hot_bits:
            raise ValueError("tail tier precision %s < hot tier %s; tail exists to "
                             "keep recent tokens accurate" % (fmt, out["hot"]))
        out[tier] = fmt
    return out


def vram_estimate_mib(spec: dict, tokens: int, layers: int, kv_heads: int,
                      head_dim: int) -> int:
    """粗估三层各占显存 (MiB), 给 GUI/预检提示用。cold 只计常驻窗口外的压缩后体积。"""
    per_token_per_layer = kv_heads * head_dim * 2  # K+V, bf16 字节数基准
    total = 0
    for tier in TIERS:
        fmt = spec.get(tier, "auto")
        bits = FORMAT_BITS.get(fmt) or 16
        frac = {"hot": 0.5, "tail": 0.2, "cold": 0.3}.get(tier, 0.3)
        # cold 是熵/iso 压缩表示, 再乘 ~0.75 的编码收益 (估算)
        enc = 0.75 if tier == "cold" and bits < 8 else 1.0
        total += tokens * frac * layers * per_token_per_layer * (bits / 16) * enc
    return int(total / (1 << 20))


if __name__ == "__main__":
    import sys
    for spec in sys.argv[1:] or ["hot=bf16,tail=fp16,cold=iso3", ""]:
        try:
            p = parse(spec)
            print("%-40r -> %s" % (spec, p))
        except ValueError as e:
            print("%-40r -> ERROR: %s" % (spec, e))
