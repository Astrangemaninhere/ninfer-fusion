#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adapt.py — v4 自适应控制器: 面对(全新)架构自动产出完整适配工件集.

流程 (全自动, 无手工适配; 产物可审计可回滚):
  1. spec: 输入是 HF config.json -> 走 arch_spec.hf_to_spec 抽取 (新旋钮字段);
     **输入已经是 archkit spec.json 时按原样读** (adapt_all.py 传的就是这种)。
     从前没有这一支: spec 被重新喂给 hf_to_spec(), 于是 geometry 变成空对象,
     报错在三个栈帧之后以 "MoE-only spec" 的形式出现 —— 那是误诊。
  2. 口味判定: layer_types + 权重键普查 -> flavors 匹配
  3. 算子目录缺口判定 (catalog): 每个需求 -> covered | engine_hook | new_op
     - covered      : 既有共享算子组合 (v3 gen_variant 出叶子)
     - engine_hook  : 需要引擎 constexpr 钩子 (softcap/output_multiplier/
                      qk_scale/window) -> 自动生成 patch 文本 (apply 后编译)
     - new_op       : 需一次内核实现 (模板+定位给出, 之后进 flavors 全自动)
     层型序列读不出来时**不判定缺口**, 清单写成 `"gaps": null` +
     `"gaps_measured": false` (读者渲染成"未测量"), 因为"每一层都当 full
     attention"会让混合模型的 GDN 缺口整批消失。
     同一种输入下 config.h **拒绝写**: 层型序列读不出来时, 三个层型计数
     (full/swa/gdn) 只能全写 0 —— 一个 33 层的模型宣称"0 层是任何一类",
     而叶子就是按这张表切片的。以前它照写不误、rc=0; 现在指名拒绝、rc=2,
     并且不覆盖上一次运行留下的 config.h (那份头文件的层型表属于另一次输入)。
  4. 输出: config.h / leaves / bindings.stub / engine_patch.h.diff /
     manifest (缺口清单与验收路径)

用法: python adapt.py <model_dir_or_config.json_or_spec.json> [--family auto]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_spec  # noqa: E402
import flavors    # noqa: E402
import gen_variant  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# §123 (1) 回归恢复: model_id -> 合法 C++ 标识符
# ---------------------------------------------------------------------------
def cpp_ident(model_id) -> str:
    """model_id -> 合法 C++ 标识符 (_TODO.md §123 第 1 条, 逐字依据).

    生成器当时只把 '-' 换成 '_'，没有处理 '.'：`spark-x2.5-4b` 于是产出
    `namespace ninfer::targets::spark_x2.5_4b::detail {` —— 一个编不过的头文件，
    却在 rc=0 下被写出来。发现方式是**读产物第 4 行**，不是读生成器；所以它活了
    整整一轮（§123: "生成的 config.h 编不过 ... 阻塞级"）。
    规则: 所有非 [0-9A-Za-z_] 字符折成 '_'；首字符是数字则前缀 '_'。
    """
    ident = re.sub(r'[^0-9A-Za-z_]', '_', str(model_id))
    if not ident or ident[0].isdigit():
        ident = '_' + ident
    return ident


# ---------------------------------------------------------------------------
# §123 (2) 的方法: 探测器读**引擎自己的注册表/算子目录**, 不镜像一份常量表。
# 后果: A3 落地的三项 new_op 在这里自动变成 covered, 不需要再手工改这张表一次。
# ---------------------------------------------------------------------------
def _engine_text(rel_path):
    try:
        return (REPO / rel_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def engine_has(rel_path, pattern) -> bool:
    text = _engine_text(rel_path)
    return bool(text is not None and re.search(pattern, text))


def engine_registers_geometry(q_heads, kv_heads, head_dim) -> str:
    """命中 `using GqaXxx = GqaGeometry<q, kv, split[, head_dim]>` 的别名名, 否则 ''.

    head_dim 缺省 256 (GqaGeometry 的默认模板参数) —— 不处理它会把
    `GqaGeometry<16, 4, 1>` 错判成 head_dim 未知。
    """
    text = _engine_text('src/ops/kernel/gqa_attention_geometry.cuh')
    if text is None:
        return ''
    pat = re.compile(r'using\s+(Gqa\w+)\s*=\s*GqaGeometry<\s*(\d+)\s*,\s*(\d+)\s*,'
                     r'\s*\d+\s*(?:,\s*(\d+)\s*)?>')
    for m in pat.finditer(text):
        name, q, kv, hd = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        hd = int(hd) if hd else 256
        if (q, kv, hd) == (int(q_heads), int(kv_heads), int(head_dim)):
            return name
    return ''

#: Keys only an HF config.json carries. A JSON object that has `geometry` but none of
#: these is an archkit spec that was already extracted -- running it back through
#: hf_to_spec() is what produced the two `"geometry": {}` specs in this tree
#: (specs/qwen4-exp_spec.json, specs/specs_spec.json), after which every consumer read
#: its own default and the pipeline died inside emit_config_header() blaming MoE.
_HF_CONFIG_KEYS = ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                   'num_key_value_heads', 'vocab_size', 'intermediate_size',
                   'model_type', 'architectures', 'text_config')


def looks_like_spec(cfg) -> bool:
    """True: an already-extracted archkit spec. False: an HF config.json."""
    if not isinstance(cfg, dict):
        return False
    tc = cfg.get('text_config') or cfg
    if any(k in tc for k in _HF_CONFIG_KEYS):
        return False
    return 'geometry' in cfg


def spec_from_spec(cfg: dict, path) -> dict:
    """Take an already-extracted spec as it is -- never re-extract it.

    adapt_all.py hands this program a `specs/<id>_spec.json`, so this is the normal
    entry, not an edge case: re-extracting produced `geometry: {}` and the failure
    surfaced three frames later as "MoE-only spec must say so on purpose", which is a
    misdiagnosis (nothing here is a MoE-only model; the geometry is simply gone).
    """
    if not cfg.get('geometry'):
        raise SystemExit(
            "%s is an archkit spec whose 'geometry' object is EMPTY, so there is nothing "
            "to translate. That shape is what hf_to_spec() leaves behind when it is "
            "handed a spec instead of a config.json -- this file was produced by that "
            "round trip. Remedy: re-extract from the model's config.json, e.g. "
            "python3 %s/adapt_all.py <model folder>."
            % (path, REPO / 'tools' / 'archkit'))
    spec = dict(cfg)
    # adapt.py's own readers (catalog_gaps / emit_config_header) read
    # `layer_kind_order`, while specs written by hf_to_spec()/gguf_spec.py carry
    # `layer_types`. Reading only one of the two is a SILENT all-Full default: every
    # GDN layer of a hybrid model gets booked as full attention, the
    # `attention:linear(gdn)` new_op gap disappears, and the pipeline calls a model the
    # engine cannot run servable. Both spellings, or nothing.
    spec['layer_kind_order'] = list(spec.get('layer_kind_order')
                                    or spec.get('layer_types') or [])
    g = spec['geometry']
    knobs = dict(spec.get('knobs') or {})
    if knobs.get('post_norm_eps') is None and g.get('rms_eps') is not None:
        knobs['post_norm_eps'] = g['rms_eps']
    spec['knobs'] = knobs
    spec.setdefault('multimodal', False)
    spec.setdefault('rope', {})
    spec['head_dim'] = spec.get('head_dim') or g.get('head_dim')
    return spec


def layer_kind_order(spec) -> list:
    """规范化的层型序列: 两个拼法都认, 认不出就返回空 (调用方必须拒绝猜)。"""
    return list(spec.get('layer_kind_order') or spec.get('layer_types') or [])


def gaps_are_measured(spec) -> bool:
    """能不能真的判定缺口?

    判据: 层型序列必须读得到。读不到时 catalog_gaps() 会把**每一层**当 full
    attention (它只按 Counter 里出现的键判定), 于是混合模型里的 GDN 层会整批
    消失、`attention:linear(gdn)` 这个 new_op 缺口不会出现 —— 那是一个"看起来
    测过了、其实没测"的绿灯。所以这种情况下的产物必须写 gaps: null +
    gaps_measured: false, 让读者渲染成"未测量"。
    """
    if not spec['geometry'].get('layers'):
        return False
    return bool(layer_kind_order(spec))


def extract_spec(config_path, model_id):
    """带新旋钮的规格抽取 (输入必须是 HF config.json)。"""
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    tc = cfg.get('text_config') or cfg
    spec = arch_spec.hf_to_spec(config_path, model_id, 'auto',
                                out=os.path.join(REPO, 'tools', 'archkit', 'specs',
                                                 model_id + '_spec.json'))
    knobs = {}
    # §121 (1): 白名单漏了这五个，后两个探测器就**永远不会触发** ——
    # "knobs 白名单补齐" 是那次修复的第一条，不是可选项。
    for k in ('sliding_window', 'qk_scale_factor', 'output_multiplier',
              'final_logit_softcapping', 'layer_rope_theta', 'post_norm_eps',
              'attention_bias', 'tie_word_embeddings',
              'partial_rotary_factor', 'headwise_attn_output_gate',
              'gate_attn_act_mode', 'hidden_act'):
        if k in tc:
            v = tc[k]
            if k == 'layer_rope_theta':
                v = [float(x) if x else 0.0 for x in v]
            knobs[k] = v
    spec['knobs'] = knobs
    spec['layer_kind_order'] = list(tc.get('layer_types') or [])
    spec['multimodal'] = bool(cfg.get('vision_config'))
    spec['rope'] = tc.get('rope_parameters') or {}
    spec['head_dim'] = tc.get('head_dim')
    # §121 (2): 这些必须**落盘**。hf_to_spec 只写它自己那份 dict，探测器读的是
    # 落盘后的 spec —— 不落盘 = 审计脚本重读文件看到不同 spec，"重读看到的不同"
    # 正是假绿灯的来源。
    _rope = spec['rope'] or {}
    spec['rope_by_kind'] = {k: v.get('rope_theta') for k, v in _rope.items()
                            if isinstance(v, dict) and v.get('rope_theta') is not None}
    spec['partial_rotary_by_kind'] = {k: float(v.get('partial_rotary_factor', 1.0))
                                      for k, v in _rope.items() if isinstance(v, dict)}
    spec['hidden_act'] = tc.get('hidden_act')
    _attn = dict(spec.get('attention') or {})
    _attn['headwise_attn_output_gate'] = bool(tc.get('headwise_attn_output_gate'))
    _attn['gate_attn_act_mode'] = tc.get('gate_attn_act_mode')
    _attn['qk_norm'] = tc.get('qk_norm')          # Spark 无此键 -> None -> 报 absent
    _attn['sliding_window'] = tc.get('sliding_window')
    spec['attention'] = _attn
    return spec


def catalog_gaps(spec):
    """算子目录判定 -> [(需求, 等级, 处置)]。等级: covered|hook|new_op。"""
    g = spec['geometry']
    knobs = spec.get('knobs', {})
    out = []
    kinds = Counter(layer_kind_order(spec))
    n_full = sum(kinds.get(x, 0) for x in ('full_attention', 'full'))
    n_swa = sum(kinds.get(x, 0) for x in ('sliding_attention', 'swa', 'sliding'))
    n_gdn = sum(kinds.get(x, 0) for x in ('linear_attention', 'gdn'))
    if n_full and not n_gdn and not n_swa:
        out.append(('attention:gqa_full', 'covered', 'v3 gen'))
    if n_swa:
        out.append(('attention:sliding_window(%d)' % knobs.get('sliding_window', 0),
                    'hook', 'window mask constexpr+leaf'))
    if n_gdn:
        out.append(('attention:linear(gdn)', 'new_op', 'GDN 口味内核(参考 qwen3.6)'))
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
    # §122 裁决: tie=true 由 new_op 改判 covered —— 引擎按独立 lm_head 对象加载
    # (bindings.cpp / text_context_impl.h)，物化是**转换期动作**、不需要新算子，
    # 而转换器真的实现了它 (tools/convert/common/source_map.py)。
    if knobs.get('tie_word_embeddings') is True:
        out.append(('head:tied=true', 'covered',
                    '转换期物化 lm_head = embed^T (tools/convert/common/source_map.py)'))

    # --- §121 (3) 四个探测器 + §123 (2)(5) 两个几何/规范探测器 ----------------
    _theta_by_kind = spec.get('rope_by_kind') or {}
    _prf_by_kind = spec.get('partial_rotary_by_kind') or {}
    _attn = spec.get('attention') or {}

    def _map(d):
        return '{' + ','.join('%s:%g' % (k, v) if isinstance(v, float) else '%s:%s' % (k, v)
                              for k, v in sorted(d.items())) + '}'

    # §121 (3.1): rope:per_type_theta -> hook (引擎 ops::rope 收 rotary_dim，theta 是参数)
    if len(set(_theta_by_kind.values())) > 1:
        out.append(('rope:per_type_theta%s' % _map(_theta_by_kind), 'hook',
                    '引擎 ops::rope 收 rotary_dim，theta 是参数 => 按层型传 theta 表'))
    # §121 (3.2): partial_rotary -> hook (rope.h: 0<rotary_dim<=head_dim)
    if _prf_by_kind and any(float(v) < 1.0 for v in _prf_by_kind.values()):
        out.append(('rope:partial_rotary%s' % _map(_prf_by_kind), 'hook',
                    'include/ninfer/ops/rope.h 支持 0<rotary_dim<=head_dim => 按层型传 rotary_dim'))
    # §123 (3) 逐头输出门: 当时判 new_op; A3 把逐头标量门并入 ops::sigmoid_mul
    # 之后引擎侧零缺口，所以这里读引擎而不是读一句静态结论。
    if _attn.get('headwise_attn_output_gate'):
        _mode = _attn.get('gate_attn_act_mode') or 'sigmoid'
        if engine_has('src/ops/wrapper/sigmoid_mul.cpp', r'headwise_gate_shape'):
            out.append(('attn:headwise_output_gate(%s)' % _mode, 'covered',
                        '逐头标量门已并入 ops::sigmoid_mul (sigmoid_mul.cpp headwise_gate_shape 分支)'))
        else:
            out.append(('attn:headwise_output_gate(%s)' % _mode, 'new_op',
                        'sigmoid_mul 只支持四维同形; HF 的 [n_q,T] 逐头广播需要一条新路由'))
    # §121 (3.4) / §123 (4) gated 激活: gelu(gate)*up 曾经全树无算子; A3 补了 ops::gelu_mul
    _act = spec.get('hidden_act')
    if _act and _act != 'silu':
        if (engine_has('include/ninfer/ops/gelu_mul.h', r'gelu_mul')
                and engine_has('src/CMakeLists.txt', r'ops/wrapper/gelu_mul\.cpp')):
            out.append(('mlp:act=%s(gated)' % _act, 'covered',
                        'ops::gelu_mul(gate,up,out) 已存在且登记进构建 (ops/wrapper/gelu_mul.cpp)'))
        else:
            out.append(('mlp:act=%s(gated)' % _act, 'new_op',
                        'ops::gelu 是 in-place 单元激活; gelu(gate)*up 全树无算子'))
    # §123 (5) qk-norm 缺失: 家族**无条件** rmsnorm(q,k)，Spark 没有 qk-norm => 需门
    if _attn.get('qk_norm') in (None, 'none', 'absent') and \
            engine_has('src/targets/qwen3_6/impl/runtime/text_context_impl.h', r'rmsnorm\('):
        out.append(('attn:qk_norm=absent', 'hook',
                    '家族无条件 rmsnorm(q,k) (text_context_impl.h); 需要 qk_norm_enabled() 门'))
    # §123 (2) 头几何: 直接解析别名表比对 (q, kv, head_dim)，未注册即 new_op
    _q, _kv, _hd = g.get('query_heads'), g.get('kv_heads'), (spec.get('head_dim') or g.get('head_dim'))
    if _q and _kv and _hd:
        _alias = engine_registers_geometry(_q, _kv, _hd)
        if _alias:
            out.append(('attn:head_geometry(%dq/%dkv@%d)' % (_q, _kv, _hd), 'covered',
                        '引擎已注册 %s (gqa_attention_geometry.cuh) 且分派守卫已覆盖' % _alias))
        else:
            out.append(('attn:head_geometry(%dq/%dkv@%d)' % (_q, _kv, _hd), 'new_op',
                        'gqa_attention.cpp 按 q_heads 反推 KV 头 => 16Q/4KV 会落进 16Q/2KV 实例'))
    if spec.get('multimodal'):
        out.append(('vision', 'new_op', '视觉塔(文本先行可跳过)'))
    return out


def cpp_float(value) -> str:
    """value -> a valid C++ float literal (always an explicit '.' + 'f').

    `'%.6gf' % 1.0` is `'1f'`, and `'1f'` is not a literal: g++ looks for a
    user-defined `operator""f`.  Measured on the Spark header, line 15/16/17 --
    i.e. the "config.h 编不过" blocker §123 recorded has TWO independent causes,
    and the format string is the one that survives fixing the namespace.
    """
    return '%.6f' % float(value) + 'f'



def emit_config_header(spec) -> str:
    g = spec['geometry']
    knobs = spec.get('knobs', {})
    # See gen_full_target.py: a geometry without `intermediate` is a MoE-only model,
    # and `g.get('intermediate', 0)` would emit a TextConfig claiming a zero-wide
    # dense FFN (the `= None` variant of this bug is what shipped in
    # src/targets/qwen4_exp/impl/config.h).  Name the absence instead of inventing a
    # width for the leaves that slice with TextConfig::intermediate.
    if 'intermediate' not in g:
        raise ValueError(
            "%s: geometry has no 'intermediate' (dense FFN width).  A MoE-only spec "
            "must say so on purpose - emit -1 (no dense path) or read "
            "moe.moe_intermediate_size; a missing key is not a width."
            % spec.get('model_id'))
    kinds = layer_kind_order(spec)
    n = g.get('layers', 0)
    # An unreadable layer-kind sequence must not become a header.  This used to fall
    # back to `kind_map = ['Full'] * n` -- which was dead code, because the three
    # counts emitted below are computed from `kinds`, not from `kind_map` -- so a spec
    # with no layer kinds produced a config.h that said `static constexpr int layers =
    # 33;` next to `full_attention_layers() { return 0; }`, `swa_attention_layers() {
    # return 0; }` and `gdn_layers() { return 0; }`.  Three zero-width classes for a
    # 33-layer model: a header that cannot describe any model, written silently, rc=0
    # (measured on a spec with neither layer_types nor layer_kind_order -- see the
    # seam-4 negative control).  Unknown is not zero, and this header is what the
    # leaves slice with, so name the absence instead of inventing a table.
    if not kinds:
        raise ValueError(
            "%s: the layer-kind sequence is unreadable (the spec carries neither "
            "'layer_types' nor 'layer_kind_order'), so a config.h would have to claim "
            "a full/swa/gdn split that nobody measured -- the previous behaviour was "
            "to write one with all three counts ZERO for a %s-layer model.  Refusing "
            "to write the header: re-extract the spec (adapt_all.py <model folder>, "
            "or gguf_spec.py <model.gguf>) so it carries a layer-kind sequence."
            % (spec.get('model_id'), n))
    if n and len(kinds) != n:
        # Same accounting rule as gguf_spec's: a kind table that does not cover every
        # layer would silently leave the tail unaccounted for in the three counts.
        raise ValueError(
            "%s: geometry.layers=%d but the layer-kind sequence has %d entries; the "
            "emitted full/swa/gdn counts would not describe the model.  Both numbers "
            "must agree before a TextConfig is written."
            % (spec.get('model_id'), n, len(kinds)))
    swa_theta = '0' if not knobs.get('sliding_window') else str(knobs['sliding_window'])
    lines = [
        '// Generated by tools/archkit/adapt.py (v4 auto-adaptation)',
        '#pragma once',
        # §123 (1): cpp_ident(), NOT replace('-', '_') -- the '.' of
        # 'spark-x2.5-4b' is what made the header uncompilable.
        'namespace ninfer::targets::%s::detail {' % cpp_ident(spec['model_id']),
        'struct TextConfig {',
        '    static constexpr int hidden = %d;' % g.get('hidden', 0),
        '    static constexpr int layers = %d;' % n,
        '    static constexpr int query_heads = %d;' % g.get('query_heads', 0),
        '    static constexpr int kv_heads = %d;' % g.get('kv_heads', 0),
        '    static constexpr int head_dim = %d;' % (spec.get('head_dim') or g.get('head_dim', 0)),
        '    static constexpr int intermediate = %d;' % g.get('intermediate', 0),
        '    static constexpr int vocab = %d;' % g.get('vocab', 0),
        '    static constexpr int max_ctx = %d;' % g.get('max_ctx', 0),
        '    static constexpr float rms_epsilon = %s;'
        % cpp_float(knobs.get('post_norm_eps', 1e-5)),
        '    static constexpr int sliding_window = %s;' % swa_theta,
        '    static constexpr float qk_scale_factor = %s;'
        % cpp_float(knobs.get('qk_scale_factor', 1.0)),
        '    static constexpr float output_multiplier = %s;'
        % cpp_float(knobs.get('output_multiplier', 1.0)),
        '    static constexpr float final_logit_softcapping = %s;'
        % cpp_float(knobs.get('final_logit_softcapping', 0.0)),
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
    ap.add_argument('input', help='HF 模型目录 / config.json, 或已抽取的 archkit spec.json')
    ap.add_argument('--model-id', default='')
    args = ap.parse_args()
    cfg_path = args.input
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, 'config.json')
    model_id = args.model_id or os.path.basename(os.path.dirname(cfg_path)) \
        or 'model'
    model_id = model_id.lower().replace('_', '-')
    # THIS is the branch that used to be missing: adapt_all.py always passes an
    # already-extracted spec, and handing that back to hf_to_spec() silently emptied
    # `geometry` (see spec_from_spec). Read the file ONCE, and say plainly which of
    # the two it is instead of re-deriving the spec from itself.
    try:
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise SystemExit('%s does not exist (expected a model folder, a config.json, '
                         'or an archkit spec.json)' % cfg_path)
    except (OSError, ValueError) as exc:
        raise SystemExit('%s cannot be read as JSON: %s (a truncated download is the '
                         'usual cause)' % (cfg_path, exc))
    if looks_like_spec(cfg):
        spec = spec_from_spec(cfg, cfg_path)
        print('input is an archkit spec (geometry read as-is, NOT re-extracted)')
    else:
        spec = extract_spec(cfg_path, model_id)
    measured = gaps_are_measured(spec)
    gaps = catalog_gaps(spec) if measured else []
    out_dir = Path(__file__).resolve().parent / 'out' / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build the header BEFORE writing anything.  A refusal here (unreadable layer
    # kinds, or a geometry with no dense FFN width) must not leave a config.h behind:
    # writing it first and validating afterwards is how a header that describes nothing
    # reaches a consumer.  The stale-file note matters for the same reason -- "we did
    # not write it" and "nothing is there to be read" are different claims, and only
    # the first one is true after a refusal.
    #
    # The manifest and the hook patch are still written, and the exit code is 2: the
    # refusal is a non-zero outcome (an exit code of 0 next to "I did not produce the
    # artifact" is the same defect as BLOCKED exiting 0), while the manifest keeps the
    # "gaps unmeasured" fact readable by tools/gui/model_import --gaps, whose message
    # for it already names this generator as the way to get a real gap list.
    refused = ''
    try:
        header = emit_config_header(spec)
    except ValueError as exc:
        header = None
        refused = str(exc)
        print('adapt.py: REFUSED to write %s/config.h: %s' % (out_dir, refused))
        if (out_dir / 'config.h').exists():
            print('   !! %s/config.h exists from an EARLIER run and was NOT replaced '
                  '(its layer-kind table is from that run, not from this spec) -- do '
                  'not consume it.' % out_dir)
    # §120 O1 / §123 恢复的扣留: 有 new_op 缺口时 config.h **不许无条件写出**。
    # 缺陷原文 (adapt.py 自己的注释, 现在这句是唯一残迹): 下游只要忽略
    # manifest.gaps 就能"假装成功" —— 头文件在那儿, 于是"看起来可导入"。
    # 形态: 改名扣留为 config.h.BLOCKED + 文件自身一行 #error (覆盖"读者只看过
    # 这个文件"的情形); 退出码仍由 header_written 决定 (rc=2 = 产物集不完整)。
    # 与 §120 原文的唯一偏差, 明写不自创: **不 unlink 上一次留下的 config.h** ——
    # 本文件已把这条纪律写在上面 ("我们没写它"和"那里什么都没有"是两个不同断言),
    # unlink 会让第二个断言变真。
    _blocked = [need for need, tier, _ in gaps if tier == 'new_op'] if measured else []
    if header is not None and _blocked:
        (out_dir / 'config.h.BLOCKED').write_text(
            header + '\n#error "auto-adapt: unresolved new_op gaps (%s) -- config.h '
                     'withheld as config.h.BLOCKED; the engine has no kernel/binding for '
                     'these yet"\n' % '; '.join(_blocked), encoding='utf-8')
        refused = 'unresolved new_op gaps: %s' % '; '.join(_blocked)
        print('adapt.py: WITHHELD %s/config.h as config.h.BLOCKED: %s' % (out_dir, refused))
        header = None
    if header is not None:
        (out_dir / 'config.h').write_text(header, encoding='utf-8')
    (out_dir / 'engine_hook.patch').write_text(emit_engine_hook_patch(spec, gaps),
                                               encoding='utf-8')
    # `gaps` is a LIST only when the gaps were actually measured. A spec whose layer
    # kinds could not be read is written as `"gaps": null` + `"gaps_measured": false`,
    # which the reader (tools/gui/model_import.manifest_gap_state) renders as
    # UNMEASURED -- never as "no gaps".
    manifest = {'model_id': model_id, 'gaps_measured': measured, 'spec': spec}
    if measured:
        manifest['gaps'] = [{'need': a, 'tier': b, 'action': c} for a, b, c in gaps]
    else:
        manifest['gaps'] = None
        manifest['gaps_source'] = (
            'none: could not decide the layer kinds (the spec carries neither '
            'layer_types nor layer_kind_order), so which layers need a new kernel is '
            'UNKNOWN -- not "nothing is missing"')
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print('adapted %s -> %s' % (model_id, out_dir))
    for need, tier, action in gaps:
        print('  [%s] %-46s %s' % (tier, need, action))
    if not measured:
        print('  !! GAPS NOT MEASURED: 这份 spec 没有层型序列, 无法判定哪些层缺算子。')
        print('     manifest.json 写成 "gaps": null + "gaps_measured": false, 读者会报'
              '"无法判定" —— 不要把它当成"没有缺口"。')
    if refused:
        print('== REFUSED: config.h was not written (%s); manifest.json records the '
              'unmeasured gaps. exit 2 = the artifact set is incomplete.' % refused)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
