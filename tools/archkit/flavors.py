#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flavors.py — 叶子口味声明库 (ARCHKIT v3 核心).

特化 = 共享算子按 constexpr 几何的组合。每种口味声明"算子序列 + 行切分",
生成器据此直接产出 variant 叶子代码 —— 与手写 qwen3_6_27b 叶子同构
(同一批内核 + 编译期闭合 => 性能等同手写, 绝非通用解释层)。

口味覆盖矩阵 (新增架构 = 选口味 + 提供权重映射, 无手工 C++):

attention:
  qwen38_full   : qkv(gate)融合投影 + q_norm/k_norm + rope + gqa_attn +
                  sigmoid(gate) 乘 + o_proj
  qwen2_full    : q/k/v 独立投影 + qk_norm(可选) + rope + gqa_attn + o_proj (无 gate)
  gemma_full    : q/k/v 独立 + qk_norm 必选 + rope + gqa_attn + o_proj (无 gate)
mlp:
  swiglu_fused  : gate_up 融合 + silu + down
  swiglu_split  : gate/up 独立 + silu + down
  moe_shared    : router + experts + shared (FlashNext, 后补)
norm: rms(qwen/gemma) / layer(未来)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttnFlavor:
    name: str
    fused_qkv: bool            # True: 单矩阵含 q|k|v(+可选 gate) 行
    gate: bool                 # attention output gate (sigmoid_mul)
    qk_norm: bool
    rope: bool
    attn_kind: str = 'gqa'     # gqa | swa(full-as-gqa 里程碑) | linear
    head_split: str = 'qkv'    # 行序: qkv / qkv_gate


@dataclass(frozen=True)
class MlpFlavor:
    name: str
    gate_up_fused: bool
    act: str = 'silu'
    kind: str = 'dense'        # dense | moe


@dataclass
class LayerPattern:
    """逐层口味表 (来自 spec.layer_types/权重映射)。"""
    kinds: list = field(default_factory=list)   # per-layer: 'full'|'swa'|'gdn'|'linear'
    attn: AttnFlavor = AttnFlavor('qwen38_full', True, True, True, True)
    mlp: MlpFlavor = MlpFlavor('swiglu_fused', True)

    def __post_init__(self):
        self.attn = self.attn or AttnFlavor('qwen38_full', True, True, True, True)
        self.mlp = self.mlp or MlpFlavor('swiglu_fused', True)


FLAVORS = {
    'qwen38': LayerPattern(
        kinds=['full' if (i + 1) % 4 == 0 else 'gdn' for i in range(64)],
        attn=AttnFlavor('qwen38_full', True, True, True, True),
        mlp=MlpFlavor('swiglu_fused', True)),
    'qwen2': LayerPattern(
        kinds=['full'] * 36,
        attn=AttnFlavor('qwen2_full', False, False, False, True),
        mlp=MlpFlavor('swiglu_fused', True)),
    'gemma4': LayerPattern(
        kinds=[],   # 由 spec.layer_types 填充 (swa/full 混合)
        attn=AttnFlavor('gemma_full', False, False, True, True),
        mlp=MlpFlavor('swiglu_split', False)),
}

# 行切分语义 (fused qkv): 返回各段行数
def qkv_row_split(attn: AttnFlavor, qh: int, kvh: int, hd: int) -> dict:
    q = qh * hd
    k = kvh * hd
    v = kvh * hd
    if attn.gate:
        return {'q': q, 'gate': q, 'k': k, 'v': v, 'total': 2 * q + k + v}
    return {'q': q, 'k': k, 'v': v, 'total': q + k + v}


def mlp_row_split(mlp: MlpFlavor, hidden: int, intermediate: int) -> dict:
    if mlp.gate_up_fused:
        return {'gate_up': 2 * intermediate, 'down': hidden}
    return {'gate': intermediate, 'up': intermediate, 'down': hidden}
