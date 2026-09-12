#!/usr/bin/env python3
"""离线复算：用 dump 出来的 16x16 分数矩阵回答三个问题

1) 自校验：布局与 -inf 模式是否符合内核约定（index = b + batch*(s + steps*(p + top_k*c))）。
2) 树材料是否真的存在（打分意义）：Viterbi 最优路径分 vs 现有贪心 walk 路径分。
3) 逐深度 hit@b 上界：target 的 argmax 在该深度候选里排第几（unary 序 / row-0 分数序）。

用法: python3 df2scores_analyze.py <call_index> [--log <run log>]
"""
import argparse
import math
import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
TOPK = 16


def load_i32(path):
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def load_f32(path):
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call", type=int)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    n = a.call

    fs = DL / f"df2scores_scores_{n}.bin"
    fc = DL / f"df2scores_cand_{n}.bin"
    fu = DL / f"df2scores_unary_{n}.bin"
    for p in (fs, fc, fu):
        if not p.exists():
            print(f"缺文件：{p}")
            return
    scores = load_f32(fs)
    cand = load_i32(fc)
    unary = load_f32(fu)

    total = len(scores)
    # scores 总元素 = batch*steps*256 ⇒ batch*steps = total/256
    cells = total // (TOPK * TOPK)
    cand_cells = len(cand) // TOPK
    print(f"call={n}  scores 元素={total} ⇒ batch*steps={cells}；cand 元素={len(cand)} ⇒ {cand_cells}")
    assert cells == cand_cells, "scores 与 cand 的 batch*steps 不一致"

    # 布局: b + batch*(s + steps*(p + 16*c))；取 batch=1 时 index = s + steps*(p + 16*c)
    # 若 batch>1 则 steps = cells/batch 未知；先按 batch=1 解析并自校验 -inf 模式
    steps = cells
    def sc(s, p, c, b=0, batch=1):
        return b + batch * (s + steps * (p + TOPK * c))
    def cd(s, c, b=0, batch=1):
        return b + batch * (s + steps * c)

    # ---- 1) 自校验：s=0 且 p>0 应为 -inf；s>0 应无 -inf（采样路径除外）----
    bad = 0
    for p in range(1, TOPK):
        v = scores[sc(0, p, 0)]
        if not math.isinf(v) or v > 0:
            bad += 1
    ninf_elsewhere = sum(1 for s in range(1, steps)
                         for p in range(TOPK) for c in range(TOPK)
                         if math.isinf(scores[sc(s, p, c)]))
    print(f"[自校验] s=0,p>0 非 -inf 的条目={bad}（应为 0）；s>=1 出现 -inf 的条目={ninf_elsewhere}（应为 0）")

    # ---- 2) 贪心 walk 路径分 vs Viterbi 最优路径分 ----
    # 贪心: previous=0; c*_s = argmax_c scores[s][previous][c]（并列取最小 rank）
    def greedy():
        prev, path, tot = 0, [], 0.0
        for s in range(steps):
            best_c, best_v = 0, -math.inf
            for c in range(TOPK):
                v = scores[sc(s, prev, c)]
                if v > best_v:
                    best_v, best_c = v, c
            path.append(best_c)
            tot += best_v
            prev = best_c
        return path, tot

    # Viterbi: V_s[c] = max_p (V_{s-1}[p] + scores[s][p][c])；s=0 只有 p=0 有效
    def viterbi():
        V = [scores[sc(0, 0, c)] for c in range(TOPK)]
        back = [[0] * TOPK]
        for s in range(1, steps):
            nV, nb = [-math.inf] * TOPK, [0] * TOPK
            for c in range(TOPK):
                bp, bv = 0, -math.inf
                for p in range(TOPK):
                    v = V[p] + scores[sc(s, p, c)]
                    if v > bv:
                        bv, bp = v, p
                nV[c], nb[c] = bv, bp
            V, _ = nV, back.append(nb)
            V = nV
            back.append(nb)
        end_c = max(range(TOPK), key=lambda c: V[c])
        # 回溯
        path = [end_c]
        for s in range(steps - 1, 0, -1):
            path.append(back[s][path[-1]])
        path.reverse()
        return path, V[end_c]

    gp, gt = greedy()
    vp, vt = viterbi()
    print(f"[贪心 walk] 路径 rank={gp}  路径分={gt:.4f}")
    print(f"[Viterbi ] 路径 rank={vp}  路径分={vt:.4f}")
    print(f"[树材料] Viterbi - 贪心 = {vt - gt:+.4f}"
          f"（若≈0 则'树材料存在'在打分意义下不成立）")

    # ---- 3) 逐深度 hit@b（需要 target argmax；从运行日志的 [df2dbg] 里取）----
    if a.log:
        lg = pathlib.Path(a.log).read_text(errors="replace")
        rounds = []
        for m in re.finditer(r"\[df2dbg\] col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", lg):
            rounds.append((int(m.group(1)), int(m.group(5))))
        by_col = {}
        for col, argmax in rounds:
            by_col.setdefault(col, []).append(argmax)
        print(f"[dbg] 读到 {len(rounds)} 条列记录，最大列={max(by_col) if by_col else -1}")
        # 用 s=0 的 draft 第一个 token 与 dbg 的 draft 值做对齐（若日志含 draft 列）
        print("  （hit@b 需要把本次 dump 的 call 对齐到某一轮：用 [df2dbg] 的 col0 draft 与 dump 的 s=0 贪心 token 匹配）")
        tokens0 = cand[cd(0, gp[0])]
        print(f"  本次 dump 的 s=0 贪心 token = {tokens0}")
        hits = [0] * steps
        for col in range(steps):
            vals = by_col.get(col)
            if not vals:
                continue
            tgt = vals[0]
            # 该列候选（unary 序）里 target 的排名
            cl = [cand[cd(col, c)] for c in range(TOPK)]
            rank = cl.index(tgt) if tgt in cl else 99
            if rank < TOPK:
                hits[col] = rank
        print("  逐列 target 在候选(unary 序)中的排名（99=不在候选中）:", hits)
        for b in (1, 2, 4, 8, 16):
            ok = sum(1 for r in hits if r < b)
            print(f"    hit@{b:<2} = {ok}/{steps} = {100.0*ok/max(steps,1):.1f}%")


if __name__ == "__main__":
    main()
