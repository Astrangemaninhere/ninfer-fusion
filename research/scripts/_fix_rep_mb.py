#!/usr/bin/env python3
"""修门控体积估算：正确公式 = 扩展成元素级同形所需的权重 = (n_q*head_dim) x hidden，
BF16 每层 n_q*hd*hidden*2 字节，再乘层数（Spark: 16*256*2560*2*36 = 755 MB，与 S54 的独立估算一致）。"""
import pathlib
import sys

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")

A = """        rep_mb = (g0.get('query_heads', 0) * g0.get('hidden', 0) * 2 * 3) / 1e6"""
B = """        # 让元素级 sigmoid_mul 等价于逐头广播，需要把 g_proj 展开成 [n_q*head_dim, hidden]
        # 并把行按 head 复制 head_dim 次；BF16 体积 = n_q*hd*hidden*2*层数。
        rep_mb = (g0.get('query_heads', 0) * g0.get('head_dim', 0) * g0.get('hidden', 0)
                  * 2 * g0.get('layers', 0)) / 1e6"""
if "n_q*hd*hidden" in src:
    print("体积公式已修")
else:
    if src.count(A) != 1:
        print("锚点不唯一: %d" % src.count(A)); sys.exit(2)
    src = src.replace(A, B)
    print("已修体积公式")
A2 = "'零算子出路=复制 g_proj 权重(约 +%.0f MB/层) 或新内核' % rep_mb))"
B2 = "'零算子出路=把 g_proj 展开成 [n_q*head_dim, hidden] 的元素级门控(约 +%.0f MB 权重) 或新内核' % rep_mb))"
if "n_q*head_dim, hidden] 的元素级门控" in src:
    print("文案已修")
else:
    assert src.count(A2) == 1, "文案锚点不唯一"
    src = src.replace(A2, B2)
    print("已修文案")
P.write_text(src, encoding="utf-8")
print("ok")
