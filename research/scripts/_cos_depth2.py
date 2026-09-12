#!/usr/bin/env python3
"""跨深度余弦（修正布局版）。

关键：proposal_hidden / proj 都是**列主序**（[dim, k]，沿 dim 连续）⇒ 深度 s 的向量 = reshape(k, dim)[s]。
前版用 a[s::7] 是错的（跨深度混合），得到的"正交"是假象。
"""
import pathlib

import numpy as np

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")


def bf16(p):
    return np.frombuffer(p.read_bytes(), dtype="<u2").astype("uint32").__lshift__(16).view("float32")


def cosmat(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ M.T


for n in (1, 3, 6, 10):
    ph_p, pr_p = DL / f"df2scores_ph_{n}.bin", DL / f"df2scores_proj_{n}.bin"
    if not ph_p.exists():
        continue
    ph = bf16(ph_p)
    pr = bf16(pr_p)
    print(f"\n===== call {n} =====")
    print(f"  ph 元素={ph.size} （期望 5120*7=35840）  proj 元素={pr.size} （期望 256*7=1792）")
    if ph.size == 5120 * 7:
        H = ph.reshape(7, 5120)
        C = cosmat(H)
        print("  proposal_hidden 跨深度 cos（行=深度 0..6）:")
        for r in C:
            print("    " + " ".join(f"{x:7.3f}" for x in r))
        idx = [(i, j) for i in range(1, 7) for j in range(1, i)]
        print(f"    深度>=1 之间 min cos = {min(C[i,j] for i,j in idx):.4f}  "
              f"| 深度0 vs 1..6 = {[round(float(C[0,j]),3) for j in range(1,7)]}")
    if pr.size == 256 * 7:
        P = pr.reshape(7, 256)
        C = cosmat(P)
        idx = [(i, j) for i in range(1, 7) for j in range(1, i)]
        print(f"  selector proj(256) 深度>=1 min cos = {min(C[i,j] for i,j in idx):.4f}  "
              f"| 深度0 vs 1..6 = {[round(float(C[0,j]),3) for j in range(1,7)]}")
