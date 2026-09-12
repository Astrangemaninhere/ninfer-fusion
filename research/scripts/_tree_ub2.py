#!/usr/bin/env python3
"""离线兑现树上界（干净版）。

用 12 组真实 16x16 分数矩阵 + dbg 的 target 真 token，对比：
  链式 EAL（上界口径，应与引擎实测 1.32 接近）
  树 b=2 / b=4（深度 0 保持单链，深度 ≥1 分支）
  全集 16（候选集天花板）
  greedy vs Viterbi 路径分差（selector 自身目标的改善空间）

⚠️ 乐观性：dbg 的 target token 是在**链式前缀**下产生的；树走不同前缀 ⇒ target token 会变
⇒ 树的结果是**上界**，必须如此标注，不可当实测接受率。
"""
import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16


def load(path, fmt):
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}{fmt}", raw))


def main():
    lg = (DL / "s3_scores.log").read_text(errors="replace")
    pos_index = {}
    cur = None
    for line in lg.splitlines():
        m = re.search(r"\[df2dbg\] row=(\d+)", line)
        if m:
            cur = {}
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            col, pos, argmax = int(m.group(1)), int(m.group(2)), int(m.group(5))
            cur[col] = argmax
            if col == 0:
                pos_index[pos] = dict(cur)
    print(f"dbg 轮可索引数 = {len(pos_index)}")

    def sc(s, steps, p, c):
        return s + steps * (p + K * c)

    def greedy_path(steps, scores):
        prev, path = 0, []
        for s in range(steps):
            bv, bc = float("-inf"), 0
            for c in range(K):
                v = scores[sc(s, steps, prev, c)]
                if v > bv:
                    bv, bc = v, c
            path.append(bc)
            prev = bc
        return path

    def viterbi_score(steps, scores):
        V = [scores[sc(0, steps, 0, c)] for c in range(K)]
        for s in range(1, steps):
            nV = [float("-inf")] * K
            for c in range(K):
                best = float("-inf")
                for p in range(K):
                    v = V[p] + scores[sc(s, steps, p, c)]
                    if v > best:
                        best = v
                nV[c] = best
            V = nV
        return max(V)

    def eval_chain(steps, scores, cand, tgt):
        prev, acc = 0, 0
        for s in range(steps):
            orders = sorted(range(K), key=lambda c: -scores[sc(s, steps, prev, c)])
            c0 = orders[0]
            if s in tgt and cand[s + steps * c0] == tgt[s]:
                acc += 1
                prev = c0
            else:
                break
        return acc

    def eval_tree(steps, scores, cand, tgt, b):
        orders0 = sorted(range(K), key=lambda c: -scores[sc(0, steps, 0, c)])
        r0 = orders0[0]
        if 0 not in tgt or cand[0 + steps * r0] != tgt[0]:
            return 0
        acc, frontier = 1, {r0}
        for s in range(1, steps):
            if s not in tgt:
                break
            width = b if s >= 1 else 1
            nxt = set()
            for p in frontier:
                for c in sorted(range(K), key=lambda c: -scores[sc(s, steps, p, c)])[:width]:
                    if cand[s + steps * c] == tgt[s]:
                        nxt.add(c)
            if not nxt:
                break
            acc = s + 1
            frontier = nxt
        return acc

    chains, t2, t4, t16, dscore, ns = [], [], [], [], [], 0
    per = []
    for n in range(1, 40):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            break
        scores = load(fs, "f")
        cand = load(DL / f"df2scores_cand_{n}.bin", "i")
        front = load(DL / f"df2scores_front_{n}.bin", "i")
        steps = len(scores) // (K * K)
        tgt = pos_index.get(front[0])
        if tgt is None:
            print(f"  call {n}: frontier={front[0]} 未在 dbg 里找到，跳过")
            continue
        ns += 1
        gp = greedy_path(steps, scores)
        gs = sum(scores[sc(s, steps, gp[s - 1] if s else 0, gp[s])] for s in range(steps))
        vs = viterbi_score(steps, scores)
        dscore.append(vs - gs)
        chains.append(eval_chain(steps, scores, cand, tgt))
        t2.append(eval_tree(steps, scores, cand, tgt, 2))
        t4.append(eval_tree(steps, scores, cand, tgt, 4))
        t16.append(eval_tree(steps, scores, cand, tgt, K))
        per.append((n, chains[-1], t2[-1], t4[-1], t16[-1], vs - gs))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"对齐样本 = {ns}")
    print(f"EAL(链, 上界口径)   = 1 + {avg(chains):.2f} = {1+avg(chains):.2f}   （引擎实测 AL = 1.32 ⇒ 口径自洽）")
    print(f"EAL(树 b=2)         = 1 + {avg(t2):.2f} = {1+avg(t2):.2f}")
    print(f"EAL(树 b=4)         = 1 + {avg(t4):.2f} = {1+avg(t4):.2f}")
    print(f"EAL(全集 16)        = 1 + {avg(t16):.2f} = {1+avg(t16):.2f}   ← 候选集天花板")
    print(f"Viterbi−贪心 路径分 = {avg(dscore):+.2f}（selector 自身目标的改善空间）")
    print()
    print("逐样本 (call, 链, b=2, b=4, 16, Δ路径分):")
    for r in per:
        print(f"  {r[0]:>2}: chain={r[1]}  b2={r[2]}  b4={r[3]}  full16={r[4]}  dscore={r[5]:+.2f}")


if __name__ == "__main__":
    main()
