#!/usr/bin/env python3
"""跨深度余弦相似度：proposal_hidden(5120) 与 selector 的 proj(256) 在深度间的区分度。

若深度 >=1 之间 cos ≈ 1 ⇒ 块内各位置的 hidden 几乎无差别 ⇒ 提案在深度间"粘住"。

注意：proj 在 dump 里是 FP32（256*7*4 字节），ph 是 BF16（5120*7*2）。
"""
import pathlib
import struct

import numpy as np

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")


def bf16(p):
    return np.frombuffer(p.read_bytes(), dtype="<u2").astype("uint32").__lshift__(16).view("float32")


def cos_matrix(rows):
    M = np.stack(rows)
    n = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return n @ n.T


for n in (1, 3, 6, 10):
    ph_p = DL / f"df2scores_ph_{n}.bin"
    pr_p = DL / f"df2scores_proj_{n}.bin"
    if not ph_p.exists():
        continue
    a = bf16(ph_p)
    ph = [a[s::7] for s in range(7)] if a.size % 7 == 0 else None
    c = np.frombuffer(pr_p.read_bytes(), dtype="<f4")
    pr = [c[s::7] for s in range(7)] if c.size % 7 == 0 else None
    print(f"\n===== call {n} =====")
    if ph:
        C = cos_matrix(ph)
        print("  proposal_hidden 跨深度 cos:")
        print("    " + "\n    ".join(" ".join(f"{x:6.3f}" for x in row) for row in C))
        idx = [(i, j) for i in range(1, 7) for j in range(1, i)]
        print(f"    深度>=1 之间的最小 cos = {min(C[i,j] for i,j in idx):.4f}")
    if pr:
        C = cos_matrix(pr)
        idx = [(i, j) for i in range(1, 7) for j in range(1, i)]
        print(f"  selector proj(256) 深度>=1 最小 cos = {min(C[i,j] for i,j in idx):.4f}")
