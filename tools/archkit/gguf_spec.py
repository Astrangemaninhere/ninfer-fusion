# -*- coding: utf-8 -*-
"""gguf_spec.py - GGUF -> archkit spec (qwen35 等家族), 打通 GGUF 模型的自动适配管线.
GGUF 没有 config.json; 本工具从 GGUF kv 元数据 (qwen35.*, llama.* 通用键) 构造
与 adapt.extract_spec 同构的 spec json, 可直接喂 adapt_all (几何门/参数门全走).
Usage: python3 gguf_spec.py <model.gguf> [--out spec.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_tensors import scan  # noqa: E402  (dims 为 u64 的成熟解析)


def read_kv(path: str) -> dict:
    """全 kv 元数据 (字符串/数值; 数组只取前几个)."""
    import struct
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def read_val(t):
        if t == 0:
            return f.read(1)[0]
        if t == 1:
            return int.from_bytes(f.read(1), 'little', signed=True)
        if t in (2, 4, 10):
            size, fmt = (2, '<H') if t == 2 else ((4, '<I') if t == 4 else (8, '<Q'))
            return struct.unpack(fmt, f.read(size))[0]
        if t in (3, 5, 11):
            size, fmt = (2, '<h') if t == 3 else ((4, '<i') if t == 5 else (8, '<q'))
            return struct.unpack(fmt, f.read(size))[0]
        if t == 6:
            return struct.unpack('<f', f.read(4))[0]
        if t == 7:
            return f.read(1)[0] != 0
        if t == 8:
            return read_str()
        if t == 9:
            atype, count = struct.unpack('<IQ', f.read(12))
            return [read_val(atype) for _ in range(count)]
        if t == 12:
            return struct.unpack('<d', f.read(8))[0]
        raise SystemExit('unknown kv type %d' % t)

    meta = {}
    for _ in range(n_kv):
        key = read_str()
        t = struct.unpack('<I', f.read(4))[0]
        meta[key] = read_val(t)
    f.close()
    return meta


# GGUF arch 前缀 -> archkit family
ARCH_FAMILY = {
    'qwen35': 'qwen38',      # llama.cpp 命名 qwen35 覆盖 qwen3.5/3.6/3.8 文本族
    'qwen3': 'qwen38',
    'qwen2': 'qwen2',
    'llama': 'qwen2',
}


def build_spec(path: str) -> dict:
    meta = read_kv(path)
    arch = str(meta.get('general.architecture', ''))
    if arch not in ARCH_FAMILY:
        raise SystemExit('未适配的 GGUF 架构: %s (已支持: %s)' % (arch, list(ARCH_FAMILY)))
    p = lambda k: meta.get('%s.%s' % (arch, k))
    tensors, _ = scan(path), None
    # 统计层kind: attn_output 存在 = full attention 层; ssm_* 存在 = linear(gdn) 层
    n_layers = int(p('block_count') or 0)
    full_layers = set()
    gdn_layers = set()
    for name, _t, _d in tensors:
        if '.attn_output.weight' in name:
            full_layers.add(int(name.split('.')[1]))
        if '.ssm_' in name:
            gdn_layers.add(int(name.split('.')[1]))
    spec = {
        'model_id': Path(path).stem.lower().replace('_', '-'),
        'family': ARCH_FAMILY[arch],
        'hf': {'source': 'gguf', 'file': path},
        'geometry': {
            'hidden': int(p('embedding_length') or 0),
            'layers': n_layers,
            'query_heads': int(p('attention.head_count') or 0),
            'kv_heads': int(p('attention.head_count_kv') or 0),
            'head_dim': int(p('attention.key_length') or 0),
            'vocab': int(p('vocab_size') or meta.get('tokenizer.ggml.tokens') and
                         len(meta['tokenizer.ggml.tokens']) or 0),
            'max_ctx': int(p('context_length') or 0),
            'intermediate': int(p('feed_forward_length') or 0),
            'rms_eps': float(p('attention.layer_norm_rms_epsilon') or 1e-6),
        },
        'layer_types': (['full_attention' if i in full_layers else
                         ('linear_attention' if i in gdn_layers else 'full_attention')
                         for i in range(n_layers)]),
        'knobs': {
            'rope_theta': float(p('rope.freq_base') or 0),
            'tie_word_embeddings': not any(n == 'output.weight' for n, _, _ in tensors),
        },
        'rope': {'rope_theta': float(p('rope.freq_base') or 0), 'rope_type': 'default'},
        'multimodal': False,
        'notes': ['spec 由 GGUF 元数据生成 (gguf_spec.py); 权重转换需 qwen35 GGUF recipe'],
    }
    # 量化格式统计 -> converter 域提示
    from collections import Counter
    qhist = Counter(t for _, t, _ in tensors)
    spec['quant_formats'] = dict(qhist)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    spec = build_spec(args.gguf)
    out = args.out or str(Path(__file__).resolve().parent / 'specs' /
                          ('%s_spec.json' % spec['model_id']))
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    print('spec ->', out)
    g = spec['geometry']
    print('geometry: %d q, %d kv, hd %d, %d layers, vocab %d, ctx %d'
          % (g['query_heads'], g['kv_heads'], g['head_dim'], g['layers'], g['vocab'],
             g['max_ctx']))
    print('layer_types: full=%d gdn=%d' %
          (spec['layer_types'].count('full_attention'),
           spec['layer_types'].count('linear_attention')))
    print('quant:', spec['quant_formats'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
