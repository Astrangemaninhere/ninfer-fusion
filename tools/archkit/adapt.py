#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adapt.py — v4 自适应控制器: 面对(全新)架构自动产出完整适配工件集.

流程 (全自动, 无手工适配; 产物可审计可回滚):
  1. 拉 config.json -> spec 抽取 (arch_spec.hf_to_spec 扩展: 新增旋钮字段)
  2. 口味判定: layer_types + 权重键普查 -> flavors 匹配
  3. 算子目录缺口判定 (catalog): 每个需求 -> covered | engine_hook | new_op
     - covered      : 既有共享算子组合 (v3 gen_variant 出叶子)
     - engine_hook  : 需要引擎 constexpr 钩子 (softcap/output_multiplier/
                      qk_scale/window) -> 自动生成 patch 文本 (apply 后编译)
     - new_op       : 需一次内核实现 (模板+定位给出, 之后进 flavors 全自动)
  4. 输出: config.h / leaves / bindings.stub / engine_patch.h.diff /
     manifest (缺口清单与验收路径)

用法: python adapt.py <model_dir_or_config.json> [--family auto]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_spec  # noqa: E402
import flavors    # noqa: E402
import semantics_v5 as sem  # noqa: E402
import topology as topo  # noqa: E402
import gen_variant  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def extract_spec(config_path, model_id):
    """带新旋钮的规格抽取。"""
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    tc = cfg.get('text_config') or cfg
    spec = arch_spec.hf_to_spec(config_path, model_id, 'auto',
                                out=os.path.join(REPO, 'tools', 'archkit', 'specs',
                                                 model_id + '_spec.json'))
    knobs = {}
    for k in ('sliding_window', 'qk_scale_factor', 'output_multiplier',
              'final_logit_softcapping', 'layer_rope_theta', 'post_norm_eps',
              'attention_bias', 'tie_word_embeddings', 'global_head_dim',
              'num_global_key_value_heads', 'attention_qk_norm', 'qk_norm'):
        if k in tc:
            v = tc[k]
            if k == 'layer_rope_theta':
                v = [float(x) if x else 0.0 for x in v]
            knobs[k] = v
    spec['knobs'] = knobs
    spec['layer_kind_order'] = list(tc.get('layer_types') or [])
    try:
        spec['semantics'] = sem.derive(tc.get('model_type', ''), knobs)
    except Exception as e:
        spec['semantics'] = {'error': str(e)}
    try:
        spec['topology'] = topo.classify(cfg, spec)
    except Exception as e:
        spec['topology'] = {'kind': 'dense', 'error': str(e)}
    spec['multimodal'] = bool(cfg.get('vision_config'))
    spec['rope'] = tc.get('rope_parameters') or {}
    spec['head_dim'] = tc.get('head_dim')
    # MoE/全键检测用: 保留完整 text_config (catalog 只认显式键会漏检 MoE)
    spec['raw_text_config'] = tc
    return spec


FAMILY_LIMITS = {
    # 共享 runtime 家族编译期上限 (与引擎 include/ninfer/types.h 等常量对齐)
    'qwen3_6': {'token_domain': 248077, 'max_layers': 64},
}


def catalog_gaps(spec):
    """算子目录判定 -> [(需求, 等级, 处置)]。等级: covered|hook|new_op。"""
    g = spec['geometry']
    knobs = spec.get('knobs', {})
    out = []
    kinds = Counter(spec.get('layer_kind_order') or [])
    n_full = sum(kinds.get(x, 0) for x in ('full_attention', 'full'))
    n_swa = sum(kinds.get(x, 0) for x in ('sliding_attention', 'swa', 'sliding'))
    n_gdn = sum(kinds.get(x, 0) for x in ('linear_attention', 'gdn'))
    if n_full and not n_gdn and not n_swa:
        out.append(('attention:gqa_full', 'covered', 'v3 gen'))
    if n_swa:
        out.append(('attention:sliding_window(%d)' % knobs.get('sliding_window', 0),
                    'hook', 'window mask constexpr+leaf'))
    if n_gdn:
        out.append(('attention:linear(gdn)', 'covered',
                    'v3 gen 叶子 (qwen3_6 引擎 GDN 主路径已有, 27b 在用)'))
    if knobs.get('qk_scale_factor'):
        out.append(('attn_scale:qk_scale_factor=%.4g' % knobs['qk_scale_factor'],
                    'hook', 'leaf 传 scale'))
    if knobs.get('output_multiplier'):
        out.append(('layer_scale:output_multiplier=%.6g' % knobs['output_multiplier'],
                    'hook', '逐层乘'))
    if knobs.get('final_logit_softcapping'):
        out.append(('head:logit_softcap=%.4g' % knobs['final_logit_softcapping'],
                    'hook', 'logits 后处理 tanh 帽'))
    if knobs.get('layer_rope_theta'):
        z = sum(1 for x in knobs['layer_rope_theta'] if x == 0)
        out.append(('rope:per_layer_theta(n=%d, zero=%d)' % (len(knobs['layer_rope_theta']), z),
                    'hook', '逐层 theta 表'))
    if knobs.get('tie_word_embeddings') is False:
        out.append(('head:tied=false', 'covered', '独立 lm_head 对象'))
    if knobs.get('tie_word_embeddings') is True:
        out.append(('head:tied=true', 'new_op', 'tied head (embed^T reuse, engine flavor)'))
    ghd = knobs.get('global_head_dim')
    gkv = knobs.get('num_global_key_value_heads')
    if ghd and ghd != g.get('head_dim'):
        out.append(('attn:hybrid_global_hd=%d(kv=%s) vs local hd=%s'
                    % (ghd, gkv, g.get('head_dim')),
                    'new_op', 'heterogeneous attention head geometry'))
    if knobs.get('attention_qk_norm') or knobs.get('qk_norm'):
        out.append(('attn:qk_norm', 'hook', 'q/k norm leaf'))
        # B 类家族编译期不变量审计 (Muse 破墙 9 门中 catalog 外的部分; 见 _AUTOADAPT.md §4B)
    fam = 'qwen3_6'  # 引擎家族归属: 目前复用的共享 runtime; spec.family 归一见 adapt_all
    limits = FAMILY_LIMITS.get(fam, {})
    vocab = g.get('vocab', 0)
    td = limits.get('token_domain')
    if td and vocab and vocab != td:
        out.append(('token_domain:vocab=%d!=family %d' % (vocab, td),
                    'hook', 'FrontendOptions.token_domain + official_specials 覆盖'))
    nlayers = g.get('layers', 0)
    ml = limits.get('max_layers', 64)
    if nlayers and nlayers > ml:
        out.append(('layers:%d>family cap %d' % (nlayers, ml),
                    'new_op', 'per-layer 数组/布局容量须按家族上限扩展'))
    elif nlayers and nlayers > 16:
        out.append(('layers:%d>16' % nlayers, 'hook',
                    'cold_slots 类 per-layer 数组容量核对 (家族已扩 64, 新家族须审计)'))
    if spec.get('multimodal'):
        out.append(('vision', 'new_op', '视觉功能(文本先走可跳过)'))
    # MoE 检测: 专家数>0 即 routed-expert 架构, 引擎无 MoE 主路径 => new_op 工作包
    n_experts = (knobs.get('num_experts')
                 or knobs.get('num_local_experts')
                 or (spec.get('raw_text_config') or {}).get('num_experts'))
    if not n_experts:
        raw = spec.get('raw_text_config') or {}
        n_experts = raw.get('num_experts') or raw.get('num_local_experts')
    if n_experts:
        per_tok = (knobs.get('num_experts_per_tok')
                   or (spec.get('raw_text_config') or {}).get('num_experts_per_tok') or '?')
        out.append(('moe:experts=%s,top=%s' % (n_experts, per_tok), 'new_op',
                    'MoE 路由专家 (swiglu expert kernel + 路由 + 专家并行/卸载)'))
    # 转换后门: 量化 linear 几何注册 (转换产出形状清单后由 recipe 校验脚本对照)
    out.append(('quant_geometry', 'post', '转换后校验: fp8/nvfp4 形状对照几何注册表 (A16 起步)'))
    return out


def emit_config_header(spec) -> str:
    g = spec['geometry']
    knobs = spec.get('knobs', {})
    kinds = spec.get('layer_kind_order') or []
    n = g.get('layers', 0)
    kind_map = []
    for k in kinds:
        if 'full' in k:
            kind_map.append('Full')
        elif 'sliding' in k:
            kind_map.append('Swa')
        else:
            kind_map.append('Gdn')
    if not kind_map:
        kind_map = ['Full'] * n
    swa_theta = '0' if not knobs.get('sliding_window') else str(knobs['sliding_window'])
    lines = [
        '// Generated by tools/archkit/adapt.py (v4 auto-adaptation)',
        '#pragma once',
        'namespace ninfer::targets::%s::detail {' % spec['model_id'].replace('-', '_'),
        'struct TextConfig {',
        '    static constexpr int hidden = %d;' % g.get('hidden', 0),
        '    static constexpr int layers = %d;' % n,
        '    static constexpr int query_heads = %d;' % g.get('query_heads', 0),
        '    static constexpr int kv_heads = %d;' % g.get('kv_heads', 0),
        '    static constexpr int head_dim = %d;' % (spec.get('head_dim') or g.get('head_dim', 0)),
        '    static constexpr int intermediate = %d;' % g.get('intermediate', 0),
        '    static constexpr int vocab = %d;' % g.get('vocab', 0),
        '    static constexpr int max_ctx = %d;' % g.get('max_ctx', 0),
        '    static constexpr float rms_epsilon = %.8gf;'
        % float(knobs.get('post_norm_eps', 1e-5)),
        '    static constexpr int sliding_window = %s;' % swa_theta,
        '    static constexpr float qk_scale_factor = %.6gf;'
        % float(knobs.get('qk_scale_factor', 1.0)),
        '    static constexpr float output_multiplier = %.8gf;'
        % float(knobs.get('output_multiplier', 1.0)),
        '    static constexpr float final_logit_softcapping = %.6gf;'
        % float(knobs.get('final_logit_softcapping', 0.0)),
    ]
    # 逐层 rope theta 表
    thetas = knobs.get('layer_rope_theta')
    if thetas:
        lines.append('    static constexpr float layer_rope_theta[%d] = {%s};'
                     % (n, ', '.join('%.1ff' % t for t in thetas[:n])))
    lines += [
        '    static constexpr int full_attention_layers() { return %d; }'
        % sum(1 for k in kinds if 'full' in k),
        '    static constexpr int swa_attention_layers() { return %d; }'
        % sum(1 for k in kinds if 'sliding' in k),
        '    static constexpr int gdn_layers() { return %d; }'
        % sum(1 for k in kinds if 'linear' in k or 'gdn' in k),
        '};',
        '} // namespace',
    ]
    return '\n'.join(lines)


def emit_engine_hook_patch(spec, gaps) -> str:
    """自动生成引擎钩子补丁 (供开发构建; 文本可审计)。"""
    knobs = spec.get('knobs', {})
    parts = ['--- engine hook patch (auto) ---', '']
    if knobs.get('final_logit_softcapping'):
        parts += [
            '// head logit softcap: 在采样/verify 前对 logits 应用 cap*tanh(x/cap)',
            '// 落点: text_context_impl.h sample/verify 路径 (搜 kCfg.final_logit_softcapping>0)',
            '// 生成代码:',
            '//   if (TextConfig::final_logit_softcapping > 0.f) {',
            '//       ops::logit_softcap(logits, TextConfig::final_logit_softcapping, s); }',
            '// 需新算子: ops::logit_softcap (元素级, 1 内核) -> flavors 收录后全自动。',
            '',
        ]
    if knobs.get('output_multiplier'):
        parts += [
            '// per-layer output multiplier: run_layers 每层叶子尾部乘 constexpr',
            '//   ops::scale(hidden, TextConfig::output_multiplier, s);',
            '',
        ]
    if knobs.get('sliding_window'):
        parts += [
            '// sliding window: full-attn 加窗掩码 (TODO-1 里程碑), 首版全窗跑通后补',
            '',
        ]
    return '\n'.join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='config.json 或含 config.json 的目录')
    ap.add_argument('--model-id', default='')
    args = ap.parse_args()
    cfg_path = args.input
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, 'config.json')
    raw = json.load(open(cfg_path, encoding='utf-8'))
    if 'model_id' in raw and 'geometry' in raw and 'text_config' not in raw and \
            'architectures' not in raw:
        # 输入已是 arch-spec: 直通, 不再二次提取 (防 spec 被当 config 自毁)
        spec = raw
        model_id = str(raw['model_id']).lower().replace('_', '-')
    else:
        model_id = args.model_id or os.path.basename(os.path.dirname(cfg_path)) \
            or 'model'
        model_id = model_id.lower().replace('_', '-')
        spec = extract_spec(cfg_path, model_id)
    gaps = catalog_gaps(spec)
    out_dir = Path(__file__).resolve().parent / 'out' / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.h').write_text(emit_config_header(spec), encoding='utf-8')
    (out_dir / 'engine_hook.patch').write_text(emit_engine_hook_patch(spec, gaps),
                                               encoding='utf-8')
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'model_id': model_id,
                   'gaps': [{'need': a, 'tier': b, 'action': c} for a, b, c in gaps],
                   'spec': spec}, f, ensure_ascii=False, indent=1)
    print('adapted %s -> %s' % (model_id, out_dir))
    for need, tier, action in gaps:
        print('  [%s] %-46s %s' % (tier, need, action))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
