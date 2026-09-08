#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compress_probe.py — ninfer artifact 无损压缩空间实测 (精度不压前提).

对 .ninfer 的每个张量对象回答:
  * 有多少字节是"结构性冗余" (零行/填充/重复 scale), 可以无损剔除?
  * code/scale 的熵 vs 当前位宽 (无损熵编码还能拿多少)?
只读, mmap 采样, 不整读大对象 (每对象抽前 N 行 + 全 scale 区)。

用法: python compress_probe.py <artifact.ninfer> [--sample-rows 64]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo')
from tools.artifact.container import Artifact  # noqa: E402


def entropy(counts: Counter, total: int) -> float:
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def probe_tensor(obj, art, sample_rows: int) -> dict:
    """对 fp8 codes+行 scale 类对象采样: 零行率/scale 熵/重复率。"""
    shape = tuple(obj.shape)
    if len(shape) < 2:
        return {}
    n, k = shape[0], shape[1]
    info = {'name': obj.name, 'shape': shape, 'bytes': obj.bytes}
    blob = art.payload(obj)
    # 行 stride = bytes / n (fp8 码 1B/元素, 但 bf16 scale 另区; 按对象布局粗分)
    row_bytes = len(blob) // max(1, n)
    sample = min(sample_rows, n)
    zero_rows = 0
    for r in range(sample):
        row = blob[r * row_bytes:(r + 1) * row_bytes]
        if not any(row):
            zero_rows += 1
    info['zero_row_frac_sample'] = round(zero_rows / sample, 4) if sample else 0
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('artifact')
    ap.add_argument('--sample-rows', type=int, default=64)
    args = ap.parse_args()

    art = Artifact.open(args.artifact)
    print('identity:', art.identity.model_id, art.identity.weights_id,
          'objects:', len(art.objects))
    tot = 0
    for o in art.objects:
        if o.kind != 'tensor':
            continue
        tot += o.bytes
    print('tensor bytes total: %.2f GB' % (tot / 1e9))

    # 只对最大的几个对象做采样 (mmap 读局部)
    tensors = sorted([o for o in art.objects if o.kind == 'tensor'],
                     key=lambda o: o.bytes, reverse=True)
    print('top objects by size:')
    for o in tensors[:8]:
        print('  %-60s %8.1f MB' % (o.name, o.bytes / 1e6))
    # 采样: 找含 scale 语义的大对象 (embedding/lm_head 通常有 padding 行)
    for o in tensors[:40]:
        if any(t in o.name for t in ('embed', 'lm_head', 'output')):
            info = probe_tensor(o, art, args.sample_rows)
            if info:
                print('probe', info)


if __name__ == '__main__':
    main()
