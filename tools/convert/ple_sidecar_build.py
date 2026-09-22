#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ple_sidecar_build.py — PLE/ngram SSD sidecar 构建器 (CPU)。

把原始 ngram 行表 (BF16 [rows, 160]) 切成 4096 对齐的物理文件并写出
ple-manifest.json (format_version 1), 格式与引擎 PleLayout::from_manifest
读取的 "Baekpica SSD-PLE sidecar" 一致 —— 这是 ngram 表"丢 SSD 用"的
落地件 (引擎侧 PleTable 分页缓存 + 同步 fault 已存在, 缺的只是造表工具)。

用法:
  python ple_sidecar_build.py --rows table.npy --out dir/
      [--ngram-size 3] [--heads-per-ngram 8] [--n-heads 16]
      [--row-dim 160] [--padded-rows 320001536] [--usable-rows 320001446]
      [--multipliers 23703573157769,20109073645365,8052911324071,0]
      [--max-file-bytes 4294967296]
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np

ROW_STRIDE_BYTES = 320        # BF16 x 160
ALIGN = 4096

MANIFEST_NAME = 'ple-manifest.json'
STAGING_SUFFIX = '.partial'      # directory-level temporary name: <out>/ple.partial


def _drop(path):
    """Remove one temporary name of OURS -- a directory or a file -- if it is there."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _write_payloads(bindir, rel, payload, n, per_file):
    """Write every payload file into `bindir`; return (files, parts, total).

    The loop below is the previous revision's, line for line, moved into a function so
    the caller can drop a staged directory when any write raises.
    """
    files, parts = [], []
    start = 0
    idx = 0
    while start < n:
        end = min(n, start + per_file)
        blob = payload[start * ROW_STRIDE_BYTES:end * ROW_STRIDE_BYTES]
        padded = (len(blob) + ALIGN - 1) // ALIGN * ALIGN
        name = 'ple-bf16-%05d-of-%05d.bin' % (idx + 1, 0)   # of 数稍后回填
        files.append({'index': idx, 'path': '%s/%s' % (rel, name),
                      'file_bytes': padded, 'payload_bytes': len(blob)})
        parts.append({'logical_part': idx, 'physical_file_index': idx,
                      'global_row_start': start,
                      'file_offset': 0, 'rows': end - start,
                      'payload_bytes': len(blob)})
        with open(os.path.join(bindir, name), 'wb') as f:
            f.write(blob)
            f.write(b'\x00' * (padded - len(blob)))
        start = end
        idx += 1
    total = idx
    for f in files:
        f['path'] = '%s/ple-bf16-%05d-of-%05d.bin' % (rel, f['index'] + 1, total)
    return files, parts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True, help='npy [n,160] bf16/f16/f32 原始行')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ngram-size', type=int, default=3)
    ap.add_argument('--heads-per-ngram', type=int, default=8)
    ap.add_argument('--n-heads', type=int, default=16)
    ap.add_argument('--row-dim', type=int, default=160)
    ap.add_argument('--padded-rows', type=int, default=320001536)
    ap.add_argument('--usable-rows', type=int, default=320001446)
    ap.add_argument('--multipliers', default='23703573157769,20109073645365,8052911324071')
    ap.add_argument('--max-file-bytes', type=int, default=4 << 30)
    args = ap.parse_args()

    rows = np.load(args.rows)
    if rows.dtype != np.float16:
        rows = rows.astype(np.float16)          # bf16 位布局在 python 里用 f16 占位
    n, d = rows.shape
    assert d == args.row_dim, 'row dimension mismatch'
    # 覆盖规则: logical parts 总和必须 == padded_vocabulary_rows; 不足补零行
    if n < args.padded_rows:
        pad = np.zeros((args.padded_rows - n, d), dtype=np.float16)
        rows = np.concatenate([rows, pad], axis=0)
        n = args.padded_rows
    payload = rows.tobytes()                     # 每行 320B, 连续
    assert len(payload) == n * ROW_STRIDE_BYTES

    os.makedirs(args.out, exist_ok=True)
    rel = 'ple'
    final_dir = os.path.join(args.out, rel)
    final_manifest = os.path.join(args.out, MANIFEST_NAME)
    # Directory-level temporary name: nothing appears under `ple/` and nothing appears at
    # the marker path until the WHOLE set -- every payload file and the manifest -- is
    # complete.  A mid-write failure therefore cannot leave payload bytes that the
    # previous run's marker still describes.  Deleting "what I wrote" instead would
    # delete the previous good artifact: a rebuild writes the same names.
    staging = final_dir + STAGING_SUFFIX
    _drop(staging)                      # ours by name; a leftover is debris from a kill
    bindir = os.path.join(staging, rel)
    os.makedirs(bindir, exist_ok=True)

    per_file = max(1, args.max_file_bytes // ROW_STRIDE_BYTES)
    retired_dir = final_dir + '.retiring'
    retired_manifest = final_manifest + '.retiring'
    marker_retired = False
    payload_published = False
    try:
        files, parts, total = _write_payloads(bindir, rel, payload, n, per_file)

        per_head_offsets = [0] * args.n_heads
        per_head_vocab = [args.usable_rows // args.n_heads] * args.n_heads
        per_head_vocab[-1] += args.usable_rows - sum(per_head_vocab)
        manifest = {
            'format_version': 1,
            'ngram_size': args.ngram_size,
            'heads_per_ngram': args.heads_per_ngram,
            'number_of_ngram_heads': args.n_heads,
            'embedding_row_dimension': args.row_dim,
            'row_stride_bytes': ROW_STRIDE_BYTES,
            'padded_vocabulary_rows': args.padded_rows,
            'usable_vocabulary_rows': args.usable_rows,
            'total_parameter_count': args.padded_rows * args.row_dim,
            'alignment_bytes': ALIGN,
            'layer_multipliers': [int(x) for x in args.multipliers.split(',')],
            'per_head_offsets': per_head_offsets,
            'per_head_vocabulary_sizes': per_head_vocab,
            'logical_parts': parts,
            'physical_files': files,
        }
        with open(os.path.join(staging, MANIFEST_NAME), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
            f.write('\n')

        # ---- publish: payload directory by rename, the manifest LAST ----
        if os.path.isdir(final_manifest) and not os.path.islink(final_manifest):
            raise IsADirectoryError(
                'refusing to publish: %s is a directory, so the previous state is '
                'already unreadable; nothing was replaced' % final_manifest)
        if os.path.lexists(final_manifest):
            # Retire the previous marker FIRST: from here on no reader can see an old
            # marker next to payloads that are being replaced.
            _drop(retired_manifest)
            os.replace(final_manifest, retired_manifest)
            marker_retired = True
        if os.path.lexists(final_dir):
            # A non-empty directory cannot be replaced, so the previous payload set
            # moves aside under a name no reader looks at.
            _drop(retired_dir)
            os.replace(final_dir, retired_dir)
        os.replace(bindir, final_dir)
        payload_published = True
        os.replace(os.path.join(staging, MANIFEST_NAME), final_manifest)
    except BaseException:
        _drop(staging)
        if marker_retired and not payload_published:
            # The payload set at the final name is still the previous one, so the
            # previous marker describes it exactly: put it back rather than leave a
            # good artifact looking incomplete.
            os.replace(retired_manifest, final_manifest)
        raise
    _drop(staging)
    _drop(retired_dir)
    _drop(retired_manifest)
    print('sidecar written: %d rows -> %d files @ %s' % (n, total, args.out))
    print('manifest fields: ngram=%d heads=%d/%d row_stride=%d usable=%d' % (
        args.ngram_size, args.heads_per_ngram, args.n_heads, ROW_STRIDE_BYTES,
        args.usable_rows))


if __name__ == '__main__':
    main()
