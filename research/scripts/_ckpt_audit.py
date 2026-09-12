#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_ckpt_audit.py -- CPU-only integrity audit for DFlash2 draft checkpoints.

Usage:  python _ckpt_audit.py data\\dflash2_ckpts\\step_006000.pt

Checks (all read-only, mmap when available so the 11GB opt state is never
fully resident):
  * strips torch.compile prefixes (_orig_mod. / _orig_mod__)
  * per-tensor TSV: name shape dtype norm isnan isinf verdict
  * hard-flags plausible-but-wrong reshapes/transposes of trainer-side names
    (fc.weight 5120x25600 vs 25600x5120; attention_conv/mlp_conv base
    2x2x320x16 vs any flat variant like 2x2x5120; gate_up 34816x5120;
    down 5120x17408; qkv 6144x5120)
  * tensor count must be 73, all 5 layers complete (14 tensors each)

Exit codes: 0 PASS; 1 NaN/Inf; 2 shape suspicion (incl. transposed);
            3 unrecognized name / count / layer completeness.
"""
import sys

import torch

DRAFT_LAYERS = 5

EXACT = {
    'fc.weight': (5120, 25600),          # nn.Linear(F_DIM=25600 -> 5120)
    'hidden_norm.weight': (5120,),
    'norm.weight': (5120,),
}

LAYER = {
    'attention_conv.base': (2, 2, 320, 16),   # engine consumes flat 2x2x5120
    'attention_conv.proj.weight': (1280, 5120),  # 2 taps * 320 groups
    'mlp_conv.base': (2, 2, 320, 16),
    'mlp_conv.proj.weight': (1280, 5120),
    'self_attn_qkv.weight': (6144, 5120),     # (32Q + 16KV) * 128
    'context_key.weight': (1024, 5120),       # 8 KV heads * 128
    'context_value.weight': (1024, 5120),
    'self_attn_o.weight': (5120, 4096),       # nn.Linear(4096 -> 5120): [out, in]
    'self_attn_q_norm.weight': (128,),
    'self_attn_k_norm.weight': (128,),
    'mlp_gate_up.weight': (34816, 5120),      # 2 * 17408
    'mlp_down.weight': (5120, 17408),
    'input_layernorm.weight': (5120,),
    'post_attention_layernorm.weight': (5120,),
}


def classify(name, shape):
    """-> (verdict, expected_shape_or_None). verdict in OK/TRANSPOSED/BAD_SHAPE/UNKNOWN."""
    if name in EXACT:
        exp = EXACT[name]
        if tuple(shape) == exp:
            return 'OK', exp
        if tuple(shape) == tuple(reversed(exp)):
            return 'TRANSPOSED', exp
        return 'BAD_SHAPE', exp
    if name.startswith('layers.'):
        parts = name.split('.')
        suffix = '.'.join(parts[2:]) if len(parts) > 3 else None
        if len(parts) == 3 and parts[2] == 'attn_conv':
            # legacy/alternative naming tolerated in the patch tooling
            pass
        if suffix in LAYER:
            exp = LAYER[suffix]
            if tuple(shape) == exp:
                return 'OK', exp
            if tuple(shape) == tuple(reversed(exp)):
                return 'TRANSPOSED', exp
            return 'BAD_SHAPE', exp
    return 'UNKNOWN', None


def strip_compile_prefix(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('_orig_mod.'):
            k = k[len('_orig_mod.'):]
        elif k.startswith('_orig_mod__'):
            k = k[len('_orig_mod__'):]
        out[k] = v
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 3
    path = sys.argv[1]
    try:
        st = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:  # older torch without mmap kwarg
        st = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:  # noqa: BLE001
        print('LOAD_FAIL %s: %s' % (path, e))
        return 3
    sd = strip_compile_prefix(st['model'] if isinstance(st, dict) and 'model' in st else st)

    print('# ckpt: %s' % path)
    print('# name\tshape\tdtype\tnorm\tisnan\tisinf\tverdict\texpected')
    problems = []
    layer_seen = {l: set() for l in range(DRAFT_LAYERS)}
    for name in sorted(sd):
        t = sd[name]
        shape = tuple(t.shape)
        verdict, exp = classify(name, shape)
        isnan = bool(torch.isnan(t.float()).any()) if t.numel() else False
        isinf = bool(torch.isinf(t.float()).any()) if t.numel() else False
        try:
            norm = '%.4g' % float(torch.linalg.vector_norm(t.float()))
        except Exception as e:  # noqa: BLE001
            norm = 'ERR:%s' % e
        print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
            name, list(shape), t.dtype, norm, isnan, isinf, verdict, exp or '-'))
        if isnan or isinf:
            problems.append((1, '%s has NaN=%s Inf=%s' % (name, isnan, isinf)))
        if verdict == 'TRANSPOSED':
            problems.append((2, '%s shape %s is the TRANSPOSE of trainer %s' % (name, shape, exp)))
        elif verdict == 'BAD_SHAPE':
            problems.append((2, '%s shape %s != trainer %s (plausible-but-wrong reshape)' % (
                name, shape, exp)))
        elif verdict == 'UNKNOWN':
            problems.append((3, 'unrecognized tensor name %s %s' % (name, shape)))
        if name.startswith('layers.') and name.split('.')[1].isdigit():
            layer_seen[int(name.split('.')[1])].add(name.split('.', 2)[-1])

    expected_total = len(EXACT) + DRAFT_LAYERS * len(LAYER)  # 3 + 5*14 = 73
    if len(sd) != expected_total:
        problems.append((3, 'tensor count %d != expected %d' % (len(sd), expected_total)))
    for l in range(DRAFT_LAYERS):
        missing = set(LAYER) - layer_seen[l]
        extra = layer_seen[l] - set(LAYER)
        if missing:
            problems.append((3, 'layers.%d missing %s' % (l, sorted(missing))))
        if extra:
            problems.append((3, 'layers.%d unrecognized %s' % (l, sorted(extra))))

    if problems:
        print('AUDIT: FAIL')
        for code, msg in problems:
            print('  [exit%d] %s' % (code, msg))
        return max(c for c, _ in problems)
    print('AUDIT: PASS (%d tensors, 5 complete layers, no NaN/Inf, all shapes exact)' % len(sd))
    return 0


if __name__ == '__main__':
    sys.exit(main())
