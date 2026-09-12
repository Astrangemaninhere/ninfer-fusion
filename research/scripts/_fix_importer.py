#!/usr/bin/env python3
"""Fix the model importer: it captured rope_parameters but had no detectors for four real
features, so `adapt.py` reported rc=0 with a *blocked-by-tied-head* verdict while silently
claiming coverage for a partial-RoPE / dual-theta / headwise-output-gate / GELU model
(XHToken/Spark-X2.5-4B). Engine facts used for the tiering (all verified in the tree):
  * ops::rope takes an explicit `rotary_dim` (include/ninfer/ops/rope.h) -> hook
  * ops::gelu exists (include/ninfer/ops/gelu.h, src/ops/wrapper/gelu.cpp) -> hook
  * ops::sigmoid_gate_mul + attn_input_proj's output_gate exist          -> hook
  * the converter hardcodes tie_word_embeddings False and the loader wants an lm_head
    object -> tied stays new_op (the importer was right there)
Idempotent: re-running is a no-op."""
import pathlib
import sys

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")

# --- (1) record the features in the spec (so the manifest/gate can act on them) ---
A1 = """    spec['rope'] = tc.get('rope_parameters') or {}
    spec['head_dim'] = tc.get('head_dim')
"""
B1 = """    spec['rope'] = tc.get('rope_parameters') or {}
    spec['head_dim'] = tc.get('head_dim')
    # Per-layer-type attention knobs. Spark-X2.5 keeps a dict keyed by layer type
    # (partial_rotary_factor + rope_theta per kind), which the flat `layer_rope_theta`
    # list above does not express, and gates its attention output per head.
    rope_params = spec['rope']
    if isinstance(rope_params, dict) and rope_params:
        spec['rope_by_kind'] = {
            str(k): {'theta': v.get('rope_theta'),
                     'partial_rotary_factor': v.get('partial_rotary_factor', 1.0)}
            for k, v in rope_params.items() if isinstance(v, dict)}
        pdims = {k: float(v.get('partial_rotary_factor') or 1.0)
                 for k, v in spec['rope_by_kind'].items()}
        if any(p != 1.0 for p in pdims.values()):
            spec['partial_rotary_by_kind'] = pdims
    if 'hidden_act' in tc:
        spec['hidden_act'] = tc['hidden_act']
    for k in ('headwise_attn_output_gate', 'gate_attn_act_mode'):
        if k in tc:
            spec['attention'][k] = tc[k]
"""
if B1.strip().splitlines()[2] not in src:
    if src.count(A1) != 1:
        print("ANCHOR1 count=%d - refusing" % src.count(A1)); sys.exit(2)
    src = src.replace(A1, B1)
    print("patched: spec now carries rope_by_kind / partial_rotary_by_kind / hidden_act / gate knobs")
else:
    print("spec part already patched")

# --- (2) the detectors ---
A2 = """    if knobs.get('tie_word_embeddings') is False:
"""
B2 = """    # Rope is per-layer-type when the config carries a kind-keyed dict: the theta differs
    # (Spark-X2.5: 5e6 full vs 1e4 sliding) and only a fraction of head_dim is rotated on
    # the full layers (partial_rotary_factor 0.25 -> rotary_dim 64 of 256). ops::rope
    # already takes rotary_dim, so both are wiring, not new kernels.
    rbk = spec.get('rope_by_kind') or {}
    if rbk:
        thetas = {k: v.get('theta') for k, v in rbk.items()}
        if len(set(str(t) for t in thetas.values())) > 1:
            out.append(('rope:per_type_theta{%s}' % ','.join(
                '%s:%s' % (k, v) for k, v in sorted(thetas.items())),
                'hook', '按层类型的 theta 表 (rope leaf)'))
        prk = spec.get('partial_rotary_by_kind') or {}
        if prk:
            parts = ','.join('%s:%g' % (k, v) for k, v in sorted(prk.items()))
            out.append(('rope:partial_rotary{%s}' % parts,
                        'hook', '按层类型 rotary_dim = head_dim * factor (ops::rope 已收 rotary_dim)'))
    elif knobs.get('partial_rotary_factor') not in (None, 1.0):
        out.append(('rope:partial_rotary=%g' % float(knobs['partial_rotary_factor']),
                    'hook', 'rotary_dim < head_dim (ops::rope 已收 rotary_dim)'))
    # Headwise attention output gate: a Linear(hidden -> heads) + sigmoid applied per head
    # to the attention output. The pieces exist (ops::sigmoid_gate_mul, attn_input_proj's
    # output_gate chunk); a new target needs the leaf wired.
    if knobs.get('headwise_attn_output_gate'):
        mode = knobs.get('gate_attn_act_mode') or 'sigmoid'
        out.append(('attn:headwise_output_gate(%s)' % mode,
                    'hook', '逐头注意力输出门控 (linear + sigmoid_gate_mul 已存在，需 leaf 接线)'))
    # MLP activation: only SiLU is on the fused swiglu path today; gelu has its own op, so
    # it is wiring, while anything else would need a new activation kernel.
    act = (spec.get('hidden_act') or '').lower()
    if act and act not in ('silu', 'swish', 'gelu', 'geglu'):
        out.append(('mlp:act=%s' % act, 'new_op', 'MLP 激活不在现有算子集'))
    elif act in ('gelu', 'geglu'):
        out.append(('mlp:act=%s' % act, 'hook', 'ops::gelu 已存在，需 MLP 前向选择激活'))
    if knobs.get('tie_word_embeddings') is False:
"""
if "rope:per_type_theta" not in src:
    if src.count(A2) != 1:
        print("ANCHOR2 count=%d - refusing" % src.count(A2)); sys.exit(2)
    src = src.replace(A2, B2)
    print("patched: four new detectors (per-type theta, partial rotary, output gate, mlp act)")
else:
    print("detectors already patched")

P.write_text(src, encoding="utf-8")
print("wrote %s (%d lines)" % (P.name, src.count("\n") + 1))
