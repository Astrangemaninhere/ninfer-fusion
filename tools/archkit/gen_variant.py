#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_variant.py — ARCHKIT v3: 规格+口味 -> 特化 Variant 叶子代码。

生成的叶子与 qwen3_6_27b 手写叶子同构: 调用同一批共享 ops
(linear/rmsnorm/rope/gqa_attention/sigmoid_mul/argmax 等), 所有几何为
constexpr -> 编译期闭合 => 性能等同手写特化。生成物:
  <out>/variant_leaves.inc   (注意力/MLP 叶子函数体, 供目标包装)

用法: python gen_variant.py <spec.json> [--family qwen38|qwen2|gemma4]
  --emit-attn / --emit-mlp 子命令输出对应叶子到 stdout (测试用)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flavors  # noqa: E402


def emit_attention_leaf(g, flavor: flavors.AttnFlavor, ns: str) -> str:
    """对齐 qwen3_6_27b 手写叶子: 共享算子组合 (行序语义在共享 op 内).
    与 Variant::attention_projection/attn_mix 同构; 几何全 constexpr。"""
    hd = g['head_dim']
    rows = [
        '    // [generated] attention leaf (family flavor %s)' % flavor.name,
        '    Tensor h = work.alloc(DType::BF16, {TextConfig::hidden, T});',
        '    ops::rmsnorm(x, *w.input_norm, TextConfig::rms_epsilon, true, h, s);',
    ]
    if flavor.fused_qkv:
        rows.append('    ops::attn_input_proj(h, fused_qkgv, q_flat, gate_flat, k_flat, '
                    'v_flat, text_policy(fused_qkgv), work, s);')
    else:
        rows.append('    // 独立 q/k/v: attn_input_proj(split 变体) 或逐 linear')
        rows.append('    ops::linear(h, *q_w, q_flat, s);')
        rows.append('    ops::linear(h, *k_w, k_flat, s);')
        rows.append('    ops::linear(h, *v_w, v_flat, s);')
    if flavor.qk_norm:
        rows.append('    ops::rmsnorm(q, *w.q_norm, TextConfig::rms_epsilon, true, qn, s);')
        rows.append('    ops::rmsnorm(k, *w.k_norm, TextConfig::rms_epsilon, true, kn, s);')
    rows.append('    ops::rope(pos, TextConfig::rotary_dim, TextConfig::rope_theta, qn, kn, s);')
    rows.append('    ops::gqa_attention(qn, kn, v, pos, valid, kv_table, kAttnScale, '
                'kv_view, envelope, work, a, s);')
    if flavor.gate:
        rows.append('    ops::sigmoid_mul(gate, a, s);')
    rows.append('    ops::linear_add(a_flat, *w.o_proj, x, text_policy(*w.o_proj), work, s);')
    return '\n'.join(rows)


def emit_mlp_leaf(g, flavor: flavors.MlpFlavor) -> str:
    rows = [
        '    // [generated] mlp tail (flavor %s)' % flavor.name,
        '    Tensor h = work.alloc(DType::BF16, {TextConfig::hidden, T});',
        '    ops::rmsnorm(x, *w.post_attn_norm, TextConfig::rms_epsilon, true, h, s);',
    ]
    if flavor.gate_up_fused:
        rows.append('    Tensor gate_up = work.alloc(DType::BF16, '
                    '{2 * TextConfig::intermediate, T});')
        rows.append('    ops::linear(h, *w.gate_up, gate_up, s);')
        rows.append('    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),')
        rows.append('                   gate_up.slice(0, TextConfig::intermediate, '
                    'TextConfig::intermediate), act, s);')
        rows.append('    ops::linear_add(act, *w.down, x, text_policy(*w.down), work, s);')
    else:
        rows.append('    Tensor g = work.alloc(DType::BF16, {TextConfig::intermediate, T});')
        rows.append('    Tensor u = work.alloc(DType::BF16, {TextConfig::intermediate, T});')
        rows.append('    ops::linear(h, *w.gate, g, s);')
        rows.append('    ops::linear(h, *w.up, u, s);')
        rows.append('    ops::silu_mul(g, u, act, s);')
        rows.append('    ops::linear_add(act, *w.down, x, text_policy(*w.down), work, s);')
    return '\n'.join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--emit', choices=['attn', 'mlp', 'all'], default='all')
    ap.add_argument('--family', default='qwen38')
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding='utf-8'))
    g = spec['geometry']
    pat = flavors.FLAVORS[args.family]
    ns = spec['model_id'].replace('-', '_')
    print('// Generated leaves for %s (family %s) — shared-op constexpr glue'
          % (spec['model_id'], args.family))
    if args.emit in ('attn', 'all'):
        print('namespace %s::detail {' % ns)
        print('void attention_leaf(...) {')
        print(emit_attention_leaf(g, pat.attn, ns))
        print('}')
        print('} // namespace')
    if args.emit in ('mlp', 'all'):
        print('void mlp_leaf(...) {')
        print(emit_mlp_leaf(g, pat.mlp))
        print('}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
