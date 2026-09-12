#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_extract.py — GGUF F32/F16/BF16 -> bf16 safetensors (qwen 家族键映射)。

供 ninfer convert.py 消费; K-quant 等其它量化 -> 明确报错换源。
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np

FMT = {0: ('f32', '<f4', 4), 1: ('f16', '<f2', 2), 2: ('bf16', '<u2', 2)}
# gguf -> HF 名
NAME_MAP = [
    ('token_embd.weight', 'model.embed_tokens.weight'),
    ('output_norm.weight', 'model.norm.weight'),
    ('output.weight', 'lm_head.weight'),
    ('blk.{i}.attn_norm.weight', 'model.layers.{i}.input_layernorm.weight'),
    ('blk.{i}.attn_norm_2.weight', 'model.layers.{i}.post_attention_layernorm.weight'),
    ('blk.{i}.attn_q.weight', 'model.layers.{i}.self_attn.q_proj.weight'),
    ('blk.{i}.attn_k.weight', 'model.layers.{i}.self_attn.k_proj.weight'),
    ('blk.{i}.attn_v.weight', 'model.layers.{i}.self_attn.v_proj.weight'),
    ('blk.{i}.attn_output.weight', 'model.layers.{i}.self_attn.o_proj.weight'),
    ('blk.{i}.ffn_gate.weight', 'model.layers.{i}.mlp.gate_proj.weight'),
    ('blk.{i}.ffn_up.weight', 'model.layers.{i}.mlp.up_proj.weight'),
    ('blk.{i}.ffn_down.weight', 'model.layers.{i}.mlp.down_proj.weight'),
]


def sk_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def bf16_bytes(arr32):
    """fp32 array -> bf16 字节 (高位截断)。"""
    return (arr32.view('<u4') >> 16).astype('<u2').tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.src, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit('not a GGUF file')
        _version, n_tensors, n_kv = struct.unpack('<IQI', f.read(16))
        for _ in range(n_kv):
            sk_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                sk_str(f)
            elif t == 6:
                f.read(4)
            elif t == 4:
                f.read(4)
            elif t in (0, 1, 2, 3, 5, 7):
                f.read(1 if t in (0, 1, 7) else 2)
            elif t in (9,):
                raise SystemExit('array metadata unsupported')
            else:
                f.read(8)
        metas = []
        for _ in range(n_tensors):
            name = sk_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            shape = struct.unpack('<%dq' % nd, f.read(8 * nd))
            typ = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            metas.append((name, tuple(shape), typ, off))
        data_off = f.tell()
        payload = {}
        for name, shape, typ, off in metas:
            if typ not in FMT:
                raise SystemExit('tensor %s type %d unsupported; 请下载 F16/BF16 版'
                                 % (name, typ))
            f.seek(data_off + off)
            nb = int(np.prod(shape)) * FMT[typ][2]
            raw = f.read(nb)
            if FMT[typ][0] == 'bf16':
                b = raw
            else:
                arr = np.frombuffer(raw, dtype=FMT[typ][1]).astype(np.float32)
                b = bf16_bytes(arr)
            hf = None
            for pat, hpat in NAME_MAP:
                if '{i}' in pat:
                    for i in range(1000):
                        if pat.format(i=i) == name:
                            hf = hpat.format(i=i)
                            break
                elif pat == name:
                    hf = hpat
                if hf:
                    break
            if hf:
                payload[hf] = (b, tuple(shape))

    os.makedirs(args.out, exist_ok=True)
    meta = {}
    offset = 0
    for k, (b, shape) in payload.items():
        meta[k] = {'dtype': 'BF16', 'shape': list(shape),
                   'data_offsets': [offset, offset + len(b)]}
        offset += len(b)
    with open(os.path.join(args.out, 'model.safetensors'), 'wb') as f:
        head = json.dumps(meta).encode()
        f.write(struct.pack('<Q', len(head)))
        f.write(head)
        for k, (b, _s) in payload.items():
            f.write(b)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump({'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3'},
                  f)
    print('extracted %d tensors -> %s' % (len(payload), args.out))


if __name__ == '__main__':
    main()
