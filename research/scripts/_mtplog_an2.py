#!/usr/bin/env python3
"""MTP 逐步 logits 离线分析（树的先决测量）。

产出：
 1) 逐深度的 top-1/top-2 logit、margin、softmax 熵 —— 熵引导逐档宽度的设计输入
 2) 轮内（按 k 分组）与全局的跨深度粘性：相邻 top-1 相同率 + 游程直方图
    （对标：dflash2 我们 56.9%/最大游程 5；参考 9.5%/最大 2）
 3) margin 的分布 —— 判断"分支该开在哪里"（margin 小才值得分支）

用法: python3 _mtplog_an.py [k]   （k = 运行时的 --draft-tokens，默认 3）
"""
import pathlib
import struct
import sys
from collections import Counter

import numpy as np

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
VOCAB = 248320
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def load_bf16(p):
    return np.frombuffer(p.read_bytes(), dtype="<u2").astype("uint32").__lshift__(16).view("float32")


def load_i32(p):
    raw = p.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def main():
    steps = sorted(int(p.stem.split("_")[-1]) for p in DL.glob("mtplg_logits_*.bin"))
    if not steps:
        print("没有 mtplg_*.bin —— 先跑一次带 NINFER_MTPLOG=1 的运行（需 --no-cuda-graph）")
        return
    print(f"dump 步数 = {len(steps)}（序号 {steps[0]}..{steps[-1]}），按 k={K} 分组")

    rows = []
    for n in steps:
        lg = load_bf16(DL / f"mtplg_logits_{n}.bin")
        tk = load_i32(DL / f"mtplg_tokens_{n}.bin")
        if lg.size != VOCAB:
            continue
        order = np.argsort(-lg)
        t1, t2 = int(order[0]), int(order[1])
        v1, v2 = float(lg[t1]), float(lg[t2])
        p_ = np.exp(lg - lg.max())
        p_ /= p_.sum()
        ent = float(-(p_ * np.log(p_ + 1e-30)).sum())
        rows.append({"n": n, "tok": tk[0] if tk else None, "top1": t1, "top2": t2,
                     "margin": v1 - v2, "ent": ent})

    print(f"\n{'步':>4} {'选定token':>10} {'top1':>9} {'top2':>9} {'margin':>8} {'熵':>7}")
    for r in rows[:27]:
        print(f"{r['n']:>4} {str(r['tok']):>10} {r['top1']:>9} {r['top2']:>9} "
              f"{r['margin']:>8.2f} {r['ent']:>7.3f}")

    mg = [r["margin"] for r in rows]
    en = [r["ent"] for r in rows]
    print(f"\nmargin: 中位 {np.median(mg):.2f}  p25 {np.percentile(mg,25):.2f}  p75 {np.percentile(mg,75):.2f}  最小 {min(mg):.2f}")
    print(f"熵:     中位 {np.median(en):.2f}  p25 {np.percentile(en,25):.2f}  p75 {np.percentile(en,75):.2f}")
    for thr in (0.5, 1.0, 2.0):
        c = sum(1 for m in mg if m < thr)
        print(f"  margin < {thr}: {c}/{len(mg)} = {100.0*c/len(mg):.1f}%（这些位置分支才可能有收益）")

    def stickiness(seq, label):
        if len(seq) < 2:
            print(f"{label}: 样本不足")
            return
        same = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i-1])
        run, runs = 1, []
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                run += 1
            else:
                runs.append(run); run = 1
        runs.append(run)
        print(f"{label}: 相邻相同率 {same}/{len(seq)-1} = {100.0*same/(len(seq)-1):.1f}%  "
              f"游程 {dict(sorted(Counter(runs).items()))}  最大 {max(runs)}")

    tops_all = [r["top1"] for r in rows]
    stickiness(tops_all, "全局（含跨轮边界，会低估粘性）")
    within = []
    for i in range(0, len(rows), K):
        grp = rows[i:i+K]
        within.extend([g["top1"] for g in grp][1:])
        within.append(None)   # 组间断点
    seq, segs = [], []
    for t in within:
        if t is None:
            if len(seq) >= 2: segs.append(seq)
            seq = []
        else:
            seq.append(t)
    tot_same = tot = 0
    maxrun = 0
    hist = Counter()
    for s in segs:
        for i in range(1, len(s)):
            tot += 1
            tot_same += (s[i] == s[i-1])
        run, runs = 1, []
        for i in range(1, len(s)):
            if s[i] == s[i-1]: run += 1
            else: runs.append(run); run = 1
        runs.append(run)
        maxrun = max(maxrun, max(runs))
        for r in runs: hist[r] += 1
    if tot:
        print(f"轮内（按 k={K} 分组）: 相邻相同率 {tot_same}/{tot} = {100.0*tot_same/tot:.1f}%  "
              f"游程 {dict(sorted(hist.items()))}  最大 {maxrun}")
    print("对标：dflash2（我们）56.9%/最大 5；参考实现 9.5%/最大 2")


if __name__ == "__main__":
    main()
