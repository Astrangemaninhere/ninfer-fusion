#!/usr/bin/env python3
"""MTP 逐步 logits 离线分析（树的先决测量）。

产出：
 1) 逐深度的 top-1/top-2 logit 值、margin、softmax 熵（熵引导宽度的设计输入）
 2) 跨深度粘性：相邻深度 top-1 token 相同率、游程（对标 dflash2 的 56.9%/最大5 与参考 9.5%/最大2）
 3) 逐深度 top-1 token 是否互异（一轮内）

注：完整 hit@b 需要"每深度的 target 真 token"，MTP 侧目前没有该日志（df2dbg 是 dflash2 专用）
⇒ 本脚本先给"分支该在哪里开"的判据；hit@b 需第二步给 MTP 侧也加 target 探针。
"""
import pathlib
import struct
from collections import Counter

import numpy as np

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
VOCAB = 248320


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
    print(f"dump 步数 = {len(steps)}（序号 {steps[0]}..{steps[-1]}）")

    rows = []
    for n in steps:
        lg = load_bf16(DL / f"mtplg_logits_{n}.bin")
        tk = load_i32(DL / f"mtplg_tokens_{n}.bin")
        if lg.size != VOCAB:
            print(f"  步 {n}: logits 元素 {lg.size} != {VOCAB}（batch>1?）跳过")
            continue
        order = np.argsort(-lg)
        t1, t2 = int(order[0]), int(order[1])
        v1, v2 = float(lg[t1]), float(lg[t2])
        p_ = np.exp(lg - lg.max())
        p_ /= p_.sum()
        ent = float(-(p_ * np.log(p_ + 1e-30)).sum())
        rows.append({"n": n, "tok": tk[0] if tk else None, "top1": t1, "top2": t2,
                     "v1": v1, "v2": v2, "margin": v1 - v2, "ent": ent})

    print(f"\n{'步':>4} {'选定token':>10} {'logits top1':>11} {'top2':>10} {'margin':>8} {'熵':>7}")
    for r in rows[:24]:
        print(f"{r['n']:>4} {str(r['tok']):>10} {r['top1']:>11} {r['top2']:>10} "
              f"{r['margin']:>8.2f} {r['ent']:>7.3f}")

    mg = [r["margin"] for r in rows]
    en = [r["ent"] for r in rows]
    print(f"\nmargin: 中位 {np.median(mg):.2f}  p25 {np.percentile(mg,25):.2f}  "
          f"p75 {np.percentile(mg,75):.2f}  最小 {min(mg):.2f}")
    print(f"熵:     中位 {np.median(en):.2f}  p25 {np.percentile(en,25):.2f}  "
          f"p75 {np.percentile(en,75):.2f}")
    small = sum(1 for m in mg if m < 1.0)
    print(f"margin < 1.0 的比例 = {small}/{len(mg)} = {100.0*small/len(mg):.1f}%"
          f"（这些位置分支才可能有用）")

    # 跨深度粘性：按 k 分组（k 由用户运行时的 --draft-tokens 决定）；这里先按相邻比较
    same = sum(1 for i in range(1, len(rows)) if rows[i]["top1"] == rows[i-1]["top1"])
    print(f"\n相邻步 top1 相同率 = {same}/{len(rows)-1} = "
          f"{100.0*same/max(len(rows)-1,1):.1f}%")
    run, runs = 1, []
    for i in range(1, len(rows)):
        if rows[i]["top1"] == rows[i-1]["top1"]:
            run += 1
        else:
            runs.append(run); run = 1
    runs.append(run)
    print(f"游程直方图 = {dict(sorted(Counter(runs).items()))}  最大游程 = {max(runs)}")
    print("（对标：dflash2 我们 56.9%/最大5；参考 9.5%/最大2）")
    print("\n⚠️ 相邻步跨轮处会比较到"上一轮的最后一步 vs 本轮第一步"，"
          "严格版应按 k 分组后只在轮内比较 —— 需要知道 k（运行时的 --draft-tokens）。")


if __name__ == "__main__":
    main()
