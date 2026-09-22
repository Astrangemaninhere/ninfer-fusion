# -*- coding: utf-8 -*-
"""gen_full_target.py — 从 adapt manifest 发射"完整 27b 同形 TextConfig"。

27b config.h 是运行时契约的活模板 (ModelConfig 逐字段映射 + 检测惯用法旋钮)。
本工具把 manifest(spec+knobs+layer_types) 直译为等形 TextConfig:
  - 常量几何/eps/旋钮 (qk_scale/output_multiplier/softcap/window)
  - 逐层 kind 表 (full/swa, gdn 恒 0 = 纯 softmax 族)
  - 逐层 rope theta 表 (0 = NoPE 整层跳过, 引用推导语义)
  - 访问器全表面 (is_full/is_swa/index/layer_of_full/rope_theta_at/...)
产物自包含 (无 qwen 头依赖), g++ -fsyntax-only 可验; 内嵌 static_assert 一致性门。

用法: python gen_full_target.py <out_dir_of_manifest> [--emit-to <engine_target_dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KIND_FULL = 0
KIND_SWA = 1
KIND_GDN = 2


def cpp_float(v):
    return repr(float(v))


def cpp_int(v):
    return str(int(v))


def rotary_dim_of(spec: dict, knobs: dict, head_dim: int) -> int:
    """rotary_dim is a declared fraction of head_dim, not always head_dim itself.

    `attention.partial_rotary_factor` is declared by shipped specs
    (specs/qwen4_exp_spec.json:72 -> 0.25 of head_dim 256,
    specs/gemma4-31b_spec.json:86) and was consumed by nothing, so the emitted
    TextConfig claimed a full-width rope.  Nothing here invents a number: a factor
    is either declared or the width stays head_dim.
    """
    def scaled(factor, where):
        f = float(factor)
        if f == 1.0:
            return head_dim
        dim = f * head_dim
        if dim != int(dim):
            raise ValueError(
                '%s: partial_rotary_factor %s * head_dim %d is not integral (%s)'
                % (spec.get('model_id'), f, head_dim, where))
        return int(dim)

    # Per-kind factors (rope_parameters.<kind>.partial_rotary_factor).  The engine
    # has exactly one rotary_dim slot (TextConfig, see
    # src/targets/muse_glimmer_30b/impl/config.h:34), so a per-kind declaration the
    # single slot cannot represent must not be silently collapsed.  Two shapes of
    # that: kinds that disagree, and a kind-keyed dict where only some kinds carry
    # the key (Gemma-4's sliding_attention entry names the key's siblings but not
    # the factor, specs/gemma4-31b_spec.json:84-93 - the absent one is the HF
    # default 1.0, i.e. full-width rope on 50 of 60 layers).
    per_kind = {}
    kinds_named = 0
    # Two places spell per-kind rope parameters out: a flat spec.rope mapping
    # (what adapt.py stores as spec['rope'], adapt.py:55) and the shipped
    # gemma4-31b spec, which keeps them under spec['attention']['rope_parameters']
    # (specs/gemma4-31b_spec.json:82-95).  Read both, so neither memory of the
    # schema silently drops the declaration.
    for source in (spec.get('rope'), (spec.get('attention') or {}).get('rope_parameters')):
        if not isinstance(source, dict):
            continue
        for kind, value in source.items():
            if not isinstance(value, dict):
                continue
            kinds_named += 1
            if value.get('partial_rotary_factor') is not None:
                per_kind[str(kind)] = float(value['partial_rotary_factor'])
    if len(set(per_kind.values())) > 1 or (kinds_named and len(per_kind) < kinds_named):
        raise ValueError(
            '%s: partial_rotary_factor is declared per layer kind (%s of %d named kind(s)) but '
            'TextConfig has a single rotary_dim; a per-layer rotary table is needed before this '
            'family can be emitted.' % (spec.get('model_id'), per_kind, kinds_named))

    scalar = knobs.get('partial_rotary_factor')
    if scalar is None:
        scalar = (spec.get('attention') or {}).get('partial_rotary_factor')
    if scalar is None and len(per_kind) == 1:
        scalar = next(iter(per_kind.values()))
    if scalar is not None:
        return scaled(scalar, 'partial_rotary_factor')
    return int(knobs.get('rotary_dim', head_dim))


def gen_text_config(spec: dict, knobs: dict) -> str:
    g = spec["geometry"]
    kinds_raw = [t for t in (spec.get("layer_types") or spec.get("layer_kind_order") or [])]
    layers = int(g["layers"])
    assert len(kinds_raw) == layers, 'layer_types length %d != layers %d' % (len(kinds_raw), layers)
    # Three-way kind mapping.  The previous form was
    # `KIND_SWA if "sliding" in t else KIND_FULL`, so every third kind
    # (linear_attention / gdn / conv) was labelled a *full attention* layer: 24 of
    # Ornith-1.5-9B's 33 layers carry "linear_attention"
    # (tools/archkit/specs/ornith-1.5-9b-q4-k-m_spec.json:19-53).  The kind names
    # below are the ones the other two emitters already use (gen_target.py:38-46
    # norm_layer_kind, adapt.py:113-119 emit_config_header).
    def kind_of(t):
        t = (t or "").lower()
        if "sliding" in t:
            return KIND_SWA
        if "full" in t:
            return KIND_FULL
        return KIND_GDN

    kinds = [kind_of(t) for t in kinds_raw]
    non_softmax = sorted({t for t, k in zip(kinds_raw, kinds) if k == KIND_GDN})
    if non_softmax:
        # This emitter's TextConfig hard-codes gdn_* = 0 and gdn_layers() == 0
        # (see the emitted struct below), i.e. it only describes the pure-softmax
        # family.  For Ornith the run used to abort further down on the Muse-only
        # per-layer-theta assert ("layer_rope_theta length 0 != layers 33"), which
        # names the wrong cause: the model has no per-layer theta table at all, and
        # what actually blocks it is 24 linear layers.  Name the real blocker,
        # first - the same idiom as the 'intermediate' guard further down.
        raise ValueError(
            "%s: layer_types carry non-softmax layers %s; this emitter describes only the "
            "pure-softmax family (its TextConfig pins gdn_* = 0 and gdn_layers() == 0).  A "
            "linear/conv/SSM layer needs the shared runtime's linear-attention path wired for "
            "this variant first." % (spec.get("model_id"), non_softmax))
    n_full = sum(1 for k in kinds if k == KIND_FULL)
    n_swa = layers - n_full

    declared_theta = [float(t) for t in (knobs.get("layer_rope_theta") or [])]
    if declared_theta:
        theta_raw = declared_theta
        assert len(theta_raw) == layers, 'layer_rope_theta length %d != layers %d' % (
            len(theta_raw), layers)
        # NoPE is a *declared* per-layer property, not a universal rule: it holds
        # for specs that spell layer_rope_theta out
        # (specs/muse-glimmer-30b_spec.json:88-140 - 13 zero entries, exactly the
        # full-attention layers).  Gemma-4 keeps rope on its full layers with a
        # different theta and a 0.25 rotary fraction
        # (specs/gemma4-31b_spec.json:84-93), so the rule is enforced only where
        # the spec declares it.
        for i, (k, th) in enumerate(zip(kinds, theta_raw)):
            if k == KIND_FULL:
                assert th == 0.0, 'layer %d full but theta=%s (contradicts NoPE rule)' % (i, th)
            else:
                assert th > 0.0, 'layer %d swa but theta=%s' % (i, th)
    else:
        # No per-layer table: every layer gets the model's declared theta, which is
        # what a single-theta family means.  This derives from a declared value
        # (never invents one) and keeps the per-layer accessor surface complete.
        theta_raw = [float(knobs.get("rope_theta") or
                           (spec.get("rope") or {}).get("rope_theta") or 1e7)] * layers

    hidden = int(g["hidden"])
    # A spec whose geometry carries no `intermediate` is a MoE-only model: its FFN
    # width is `moe.moe_intermediate_size`, which is a different quantity with
    # different consumers.  Defaulting the key to 0 (or stringifying it) handed the
    # engine a TextConfig that silently claims a zero-wide dense FFN -
    # `static constexpr int intermediate = None;` in
    # src/targets/qwen4_exp/impl/config.h is the same defect spelled by an emitter
    # that rendered the missing key instead of naming it.  Name it.
    if "intermediate" not in g:
        raise ValueError(
            "%s: geometry has no 'intermediate' (dense FFN width).  A MoE-only spec "
            "must say so on purpose - emit -1 (no dense path, the MoE work package "
            "supplies the width) or read moe.moe_intermediate_size; a missing key is "
            "not a width." % spec.get("model_id"))
    intermediate = int(g["intermediate"])
    vocab = int(g.get("vocab", 0))
    max_ctx = int(g.get("max_ctx", 0))
    qh = int(g["query_heads"])
    kv = int(g["kv_heads"])
    hd = int(g.get("head_dim", spec.get("head_dim") or 0))
    rms_eps = float(g.get("rms_eps", knobs.get("rms_norm_eps", 1e-6)))
    post_eps = float(knobs.get("post_norm_eps", rms_eps))
    rope_theta = float(knobs.get("rope_theta") or
                       (spec.get("rope") or {}).get("rope_theta") or 1e7)
    rotary_dim = rotary_dim_of(spec, knobs, hd)
    window = int(knobs.get("sliding_window", 0))
    qk_scale = float(knobs.get("qk_scale_factor", 1.0))
    mult = float(knobs.get("output_multiplier", 1.0))
    softcap = float(knobs.get("final_logit_softcapping", 0.0))
    tie = bool(knobs.get("tie_word_embeddings", False))

    q_size = qh * hd
    kv_size = kv * hd
    query_proj_rows = 2 * q_size  # q+gate 融合 (qwen3.6 同款布局; 转换层决定)
    full_idx = [i for i, k in enumerate(kinds) if k == KIND_FULL]
    swa_idx = [i for i, k in enumerate(kinds) if k == KIND_SWA]

    o = []
    A = o.append
    A('// Generated by tools/archkit/gen_full_target.py  (DO NOT EDIT BY HAND).')
    A('// spec: %s' % spec.get("model_id"))
    A('// 完整 TextConfig: 27b 同形访问器表面 + 逐层 kind/theta 表 (纯 softmax 族, gdn=0).')
    A('#pragma once')
    A('')
    A('#include <array>')
    A('#include <cstdint>')
    A('')
    A('namespace ninfer::targets::%s::detail {' % spec["model_id"].replace('-', '_'))
    A('')
    A('struct TextConfig {')
    A('    static constexpr int hidden       = %s;' % cpp_int(hidden))
    A('    static constexpr int layers       = %s;' % cpp_int(layers))
    A('    static constexpr int intermediate = %s;' % cpp_int(intermediate))
    A('')
    A('    // 输出行/采样域 (转换/artifact 阶段可再按内核 padding 调整).')
    A('    static constexpr int output_rows  = %s;' % cpp_int(vocab))
    A('    static constexpr int token_domain = output_rows;')
    A('')
    A('    // 纯 softmax 族: 无 GDN/卷积/MTP. 槽位保留 0 (共享 ModelConfig 表面).')
    A('    static constexpr int gdn_conv_kernel      = 0;')
    A('    static constexpr int gdn_conv_state_width = 0;')
    A('    static constexpr int gdn_key_heads        = 0;')
    A('    static constexpr int gdn_key_head_dim     = 0;')
    A('    static constexpr int gdn_value_heads      = 0;')
    A('    static constexpr int gdn_value_head_dim   = 0;')
    A('')
    A('    static constexpr int query_heads = %s;' % cpp_int(qh))
    A('    static constexpr int kv_heads    = %s;' % cpp_int(kv))
    A('    static constexpr int head_dim    = %s;' % cpp_int(hd))
    A('    static constexpr int rotary_dim  = %s;' % cpp_int(rotary_dim))
    A('')
    A('    static constexpr float rms_epsilon      = %sF;' % cpp_float(rms_eps))
    A('    static constexpr float post_norm_eps    = %sF;' % cpp_float(post_eps))
    A('    static constexpr float rope_theta       = %sF;' % cpp_float(rope_theta))
    A('')
    A('    // 架构旋钮 (自动适配, 参考实现推导语义).')
    A('    static constexpr int   sliding_window        = %s;' % cpp_int(window))
    A('    static constexpr float qk_scale_factor       = %sF;' % cpp_float(qk_scale))
    A('    static constexpr float final_logit_softcapping = %sF;' % cpp_float(softcap))
    A('')
    A('    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;')
    A('    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;')
    A('    static constexpr int convolution_dim       = 2 * key_dim + value_dim;')
    A('    static constexpr int query_size            = query_heads * head_dim;')
    A('    static constexpr int kv_size               = kv_heads * head_dim;')
    A('    static constexpr int query_projection_rows = 2 * query_size;')
    A('')
    A('    static constexpr int mtp_layers               = 0;')
    A('    static constexpr int mtp_input_rows           = 0;')
    A('    static constexpr int mtp_attention_input_rows = 0;')
    A('    static constexpr int mtp_mlp_gate_up_rows     = 0;')
    A('')
    A('    // 逐层口味表: 0 = full, 1 = swa (sliding window). 全部层均为 softmax 注意力;')
    A('    // gdn 层数为 0 -> 共享 run_layers 的 gdn 分支恒不进入.')
    A('    static constexpr std::array<int, %d> layer_kind{%s};'
      % (layers, ', '.join(str(k) for k in kinds)))
    A('    // 逐层 rope theta; 0 = NoPE (整层无位置编码, 引用推导语义:')
    A('    //   position_embeddings if layer_rope_theta[i] else None).')
    A('    static constexpr std::array<float, %d> layer_rope_theta{%s};'
      % (layers, ', '.join('%sF' % cpp_float(t) for t in theta_raw)))
    A('    static constexpr std::array<int, %d> kFullLayers{%s};'
      % (n_full, ', '.join(str(i) for i in full_idx)))
    A('')
    A('    [[nodiscard]] static constexpr bool is_full_attention(int layer) {')
    A('        return layer_kind[static_cast<std::size_t>(layer)] == 0;')
    A('    }')
    A('    [[nodiscard]] static constexpr bool is_swa_attention(int layer) {')
    A('        return layer_kind[static_cast<std::size_t>(layer)] == 1;')
    A('    }')
    A('    [[nodiscard]] static constexpr bool qk_norm_enabled() { return true; }')
    A('    [[nodiscard]] static constexpr bool per_layer_scalar() { return false; }')
    A('    [[nodiscard]] static constexpr bool final_logit_softcapping_enabled() {')
    A('        return final_logit_softcapping > 0.0F;')
    A('    }')
    A('    [[nodiscard]] static constexpr float output_multiplier() { return %sF; }'
      % cpp_float(mult))
    A('    [[nodiscard]] static constexpr int full_attention_layers() { return %d; }' % n_full)
    A('    [[nodiscard]] static constexpr int swa_attention_layers() { return %d; }' % n_swa)
    A('    [[nodiscard]] static constexpr int gdn_layers() { return 0; }')
    A('    [[nodiscard]] static constexpr int swa_attention_index(int layer) {')
    A('        int n = 0;')
    A('        for (int i = 0; i < layer; ++i) { n += is_swa_attention(i) ? 1 : 0; }')
    A('        return n;')
    A('    }')
    A('    [[nodiscard]] static constexpr int full_attention_index(int layer) {')
    A('        int n = 0;')
    A('        for (int i = 0; i < layer; ++i) { n += is_full_attention(i) ? 1 : 0; }')
    A('        return n;')
    A('    }')
    A('    [[nodiscard]] static constexpr int gdn_index(int) { return 0; }')
    A('    [[nodiscard]] static constexpr int layer_of_full_index(int fidx) {')
    A('        return kFullLayers[static_cast<std::size_t>(fidx)];')
    A('    }')
    A('    [[nodiscard]] static constexpr float rope_theta_at(int layer) {')
    A('        return layer_rope_theta[static_cast<std::size_t>(layer)];')
    A('    }')
    A('};')
    A('')
    A('static_assert(TextConfig::full_attention_layers() == %d);' % n_full)
    A('static_assert(TextConfig::swa_attention_layers() == %d);' % n_swa)
    A('static_assert(TextConfig::gdn_layers() == 0);')
    A('static_assert(TextConfig::layer_kind.size() == %d);' % layers)
    A('static_assert(TextConfig::layer_rope_theta.size() == %d);' % layers)
    A('// full 层位 bijection: layer_of_full_index(fidx) 逐一命中 full.')
    A('static_assert([] {')
    A('    for (int f = 0; f < TextConfig::full_attention_layers(); ++f) {')
    A('        if (!TextConfig::is_full_attention(TextConfig::layer_of_full_index(f))) {')
    A('            return false;')
    A('        }')
    A('    }')
    A('    return true;')
    A('}());')
    if declared_theta:
        A('// NoPE 语义门: full 层 theta==0, swa 层 theta>0 (参考代码推导).')
        A('static_assert([] {')
        A('    for (int l = 0; l < TextConfig::layers; ++l) {')
        A('        const bool no_pe = TextConfig::rope_theta_at(l) == 0.0F;')
        A('        if (no_pe != TextConfig::is_full_attention(l)) { return false; }')
        A('    }')
        A('    return true;')
        A('}());')
    else:
        A('// NoPE 语义门不适用: 本 spec 未声明逐层 theta (全层同 theta, 由 rope_theta 展开).')
    A('')
    A('inline constexpr float kAttentionScale = %sF;' % cpp_float(qk_scale * (1.0 / (hd ** 0.5))))
    A('inline constexpr std::uint32_t kNativeContext = %s;' % cpp_int(max_ctx))
    A('')
    A('} // namespace ninfer::targets::%s::detail' % spec["model_id"].replace('-', '_'))
    A('')
    return '\n'.join(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', help='adapt 产物目录 (含 manifest.json)')
    ap.add_argument('--emit-to', default=None, help='引擎树目标目录 (src/targets/<id>/impl)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['spec']
    knobs = spec.get('knobs') or {}
    text = gen_text_config(spec, knobs)

    model_id = spec['model_id']
    text_h = out_dir / 'text_config.h'
    text_h.write_text(text, encoding='utf-8')
    print('wrote %s (%d bytes)' % (text_h, len(text)))

    if args.emit_to:
        target_dir = Path(args.emit_to)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / 'config.h').write_text(text, encoding='utf-8')
        print('wrote %s' % (target_dir / 'config.h'))
        note = {
            'text_config_emitted': True,
            'engine_path': str(target_dir / 'config.h'),
            'kinds': [KIND_FULL if k == 0 else 'swa' for k in [0]],  # informational
        }
        del note['kinds']
        mpath = out_dir / 'manifest.json'
        manifest.setdefault('engine', {})['full_text_config'] = str(target_dir / 'config.h')
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
        print('manifest updated (engine.full_text_config)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
