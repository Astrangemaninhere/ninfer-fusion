#!/usr/bin/env python3
"""Follow-up to _fix_importer.py: the detectors read `knobs`, but `knobs` is built from a
fixed key tuple that lacks the new keys -> two detectors could never fire. Also guard
spec['attention']."""
import pathlib
import sys

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")

A = """    for k in ('sliding_window', 'qk_scale_factor', 'output_multiplier',
              'final_logit_softcapping', 'layer_rope_theta', 'post_norm_eps',
              'attention_bias', 'tie_word_embeddings', 'global_head_dim',
              'num_global_key_value_heads', 'attention_qk_norm', 'qk_norm'):"""
B = """    for k in ('sliding_window', 'qk_scale_factor', 'output_multiplier',
              'final_logit_softcapping', 'layer_rope_theta', 'post_norm_eps',
              'attention_bias', 'tie_word_embeddings', 'global_head_dim',
              'num_global_key_value_heads', 'attention_qk_norm', 'qk_norm',
              # Spark-X2.5 surfaced these: without them the rope/gate detectors below
              # could never fire (the knob never reached `knobs`).
              'partial_rotary_factor', 'headwise_attn_output_gate',
              'gate_attn_act_mode', 'hidden_act'):"""
if "headwise_attn_output_gate'," in src and "gate_attn_act_mode',\n              'hidden_act')" in src:
    print("knobs tuple already extended")
else:
    if src.count(A) != 1:
        print("ANCHOR A count=%d - refusing" % src.count(A)); sys.exit(2)
    src = src.replace(A, B)
    print("extended the knobs key tuple by 4 keys")

C = """    for k in ('headwise_attn_output_gate', 'gate_attn_act_mode'):
        if k in tc:
            spec['attention'][k] = tc[k]"""
D = """    for k in ('headwise_attn_output_gate', 'gate_attn_act_mode'):
        if k in tc:
            spec.setdefault('attention', {})[k] = tc[k]"""
if C in src:
    src = src.replace(C, D)
    print("guarded spec['attention'] with setdefault")
else:
    print("attention guard already applied or anchor absent")

P.write_text(src, encoding="utf-8")
print("now %d lines" % (src.count("\n") + 1))
