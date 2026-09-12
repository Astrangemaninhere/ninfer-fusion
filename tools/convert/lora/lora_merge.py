#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lora_merge.py — LoRA adapter -> ninfer artifact 融合 (离线权重合并路线).

为什么离线合并而不是运行时 LoRA:
  引擎的权重是 NVFP4/行量化, 运行时逐层加 delta 需要每层解量化基座或
  双权重通道; 离线融合 = 逐层: 解量化 -> W' = W + s*(B@A) -> 重量化 ->
  写回 artifact (patch_dflash2 同款 ArtifactWriter 流式机制), 运行时零改动。

融合布局映射 (PEFT 键 -> ninfer artifact 对象):
  HF/PEFT 键                          ninfer 对象 (qwen3_8_27b 家族)
  model.language_model.model.layers.N.self_attn.qkv.(lora_A|lora_B)
      -> layers/{N}/attention/query_key_value   (融合 qkv, 行序 q|k|v)
  ...self_attn.o_proj                -> layers/{N}/attention/output
  ...mlp.linear_fc1 / gate_up_proj   -> layers/{N}/mlp/gate_up (融合 gate|up)
  ...mlp.linear_fc2 / down_proj      -> layers/{N}/mlp/down
  (键名以实际 adapter safetensors 为准; --keys 可打印)

尺度: s = alpha / r; alpha 缺省 = r (scale 1)。use_rslora 时 s = alpha/sqrt(r)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# artifact 对象名模板 (qwen3_8_27b 家族; 以引擎 bindings 实际对象名为准,
# TODO(LORA): 对照 bindings.cpp 回填 q/k/v/o 融合行序与 GDN in_proj_* 语义)
LAYER_OBJ = {
    'q': 'layers/{i}/attention/query',
    'k': 'layers/{i}/attention/key',
    'v': 'layers/{i}/attention/value',
    'o': 'layers/{i}/attention/output',
    'gdn_qkv': 'layers/{i}/gdn/input_projection',
    'gdn_a': 'layers/{i}/gdn/conv_a',
    'gdn_b': 'layers/{i}/gdn/conv_b',
    'gdn_z': 'layers/{i}/gdn/output_gate',
    'gdn_out': 'layers/{i}/gdn/output_projection',
    'gate': 'layers/{i}/mlp/gate',
    'up': 'layers/{i}/mlp/up',
    'down': 'layers/{i}/mlp/down',
}

# PEFT 模块键 -> (布局组, 引擎行语义); 正则顺序匹配。
# 真实键方案 (TeichAI/Qwen3.8-27B-Fable-Distill-LoRA 实测, 64 层全量):
#   base_model.model.model.language_model.layers.N.self_attn.{q,k,v,o}_proj   (16 全注意力层)
#   base_model.model.model.language_model.layers.N.linear_attn.{in_proj_qkv,
#       in_proj_a, in_proj_b, in_proj_z, out_proj}                            (48 GDN 层)
#   base_model.model.model.language_model.layers.N.mlp.{gate,up,down}_proj    (64 层)
_MODULE_RULES = [
    (re.compile(r'self_attn\.q_proj'), 'q'),
    (re.compile(r'self_attn\.k_proj'), 'k'),
    (re.compile(r'self_attn\.v_proj'), 'v'),
    (re.compile(r'self_attn\.o_proj|self_attn\.out_proj'), 'o'),
    (re.compile(r'linear_attn\.in_proj_qkv'), 'gdn_qkv'),
    (re.compile(r'linear_attn\.in_proj_a'), 'gdn_a'),
    (re.compile(r'linear_attn\.in_proj_b'), 'gdn_b'),
    (re.compile(r'linear_attn\.in_proj_z'), 'gdn_z'),
    (re.compile(r'linear_attn\.out_proj'), 'gdn_out'),
    (re.compile(r'mlp\.(gate_proj|linear_fc1)'), 'gate'),
    (re.compile(r'mlp\.(up_proj)'), 'up'),
    (re.compile(r'mlp\.(down_proj|linear_fc2)'), 'down'),
]


def classify_module(key: str):
    """返回 (kind, layer_id) 或 None。kind 见 _MODULE_RULES。"""
    m = re.search(r'layers\.(\d+)\.', key)
    if not m:
        return None
    lid = int(m.group(1))
    for pat, kind in _MODULE_RULES:
        if pat.search(key):
            return kind, lid
    return None


def lora_scale(cfg: dict) -> float:
    alpha = float(cfg.get('lora_alpha') or cfg.get('r') or 1.0)
    r = float(cfg.get('r') or 1.0)
    if cfg.get('use_rslora'):
        return alpha / (r ** 0.5)
    return alpha / r


def build_delta(keys, sd, cfg, device='cpu') -> dict:
    """收集 (obj_kind, layer) -> delta bf16 [out,in]。lora_A[r,in] lora_B[out,r]。"""
    s = lora_scale(cfg)
    grouped: dict = {}
    for k in keys:
        if not (k.endswith('.lora_A.weight') or k.endswith('.lora_B.weight')):
            continue
        cls = classify_module(k)
        if cls is None:
            continue
        kind, lid = cls
        base = k[:-len('.lora_A.weight')] if k.endswith('.lora_A.weight') else \
            k[:-len('.lora_B.weight')]
        tag = 'A' if k.endswith('.lora_A.weight') else 'B'
        key = (kind, lid)
        grouped.setdefault(key, {})[tag] = sd[k]
    deltas = {}
    for (kind, lid), ab in grouped.items():
        if 'A' not in ab or 'B' not in ab:
            print('warn: incomplete pair at', kind, lid)
            continue
        a = ab['A'].to(torch.float32)          # [r, in]
        b = ab['B'].to(torch.float32)          # [out, r]
        d = (b @ a) * s                        # [out, in]
        deltas[(kind, lid)] = d
    return deltas


def merge_matrix(base_bf16: torch.Tensor, delta: torch.Tensor, rows: slice):
    """delta [out,in] 对齐到基座矩阵的行切片 (qkv 融合: q|k|v 行序)。"""
    w = base_bf16.float()
    w[rows] = w[rows] + delta
    return w


def header_keys(path) -> list:
    """只读 safetensors 头 (文件截断时也能用): 返回键列表。"""
    import struct as _s
    with open(path, 'rb') as f:
        n = _s.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    return [k for k in hdr if k != '__metadata__']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter-dir', help='LoRA adapter 目录 (adapter_config.json + safetensors)')
    ap.add_argument('--keys', action='store_true', help='只列出 adapter 权重键并退出')
    ap.add_argument('--list-groups', action='store_true', help='列出分类后的 (kind, layer) 组')
    args = ap.parse_args()
    if not args.adapter_dir:
        ap.error('--adapter-dir required')

    d = Path(args.adapter_dir)
    cfg = json.load(open(d / 'adapter_config.json', encoding='utf-8'))
    keys = header_keys(d / 'adapter_model.safetensors')
    if args.keys:
        for k in keys[:200]:
            print(k)
        raise SystemExit(0)
    groups = {}
    for k in keys:
        cls = classify_module(k)
        if cls:
            groups.setdefault(cls, []).append(k)
    if args.list_groups:
        from collections import Counter
        c = Counter(kind for (kind, _lid) in groups)
        for kind, n in sorted(c.items()):
            print(kind, n)
        lids = sorted({lid for (_k, lid) in groups})
        print('layers covered:', lids[0], '-', lids[-1], 'count', len(lids))
        raise SystemExit(0)
    print('scale =', lora_scale(cfg), '(alpha=%s r=%s rslora=%s)' % (
        cfg.get('lora_alpha'), cfg.get('r'), cfg.get('use_rslora')))
    print('target groups:', len(groups))
