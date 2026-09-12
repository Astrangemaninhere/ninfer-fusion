#!/usr/bin/env python3
"""离线复算：用 dump 的 16x16 分数矩阵回答
1) 布局自校验；2) Viterbi 最优路径分 vs 贪心 walk 路径分（树材料在打分意义下是否存在）；
3) 逐深度 hit@b（需 --log 指到同一次运行日志，用 [df2dbg] 对齐）。

用法: python3 df2s_an.py <call_index> [--log <log>]
"""
import argparse
import math
import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16
NEG = float("-inf")


def load(path, fmt):
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}{fmt}", raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call", type=int)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    n = a.call
    fs, fc, fu = (DL / f"df2scores_{t}_{n}.bin" for t in ("scores", "cand", "unary"))
    for p in (fs, fc, fu):
        if not p.exists():
            print(f"缺文件：{p}")
            return
    scores = load(fs, "f")
    cand = load(fc, "i")
    unary = load(fu, "f")

    cells = len(scores) // (K * K)
    steps = cells  # batch=1 情形
    print(f"call={n} scores 元素={len(scores)} ⇒ batch*steps={cells}（按 batch=1 解析 steps={steps}）")

    def sc(s, p, c):
        return s + steps * (p + K * c)

    def cd(s, c):
        return s + steps * c

    # 1) 自校验
    bad0 = sum(1 for p in range(1, K) if not math.isinf(scores[sc(0, p, 0)]))
    ninf_hi = sum(1 for s in range(1, steps) for p in range(K) for c in range(K)
                  if math.isinf(scores[sc(s, p, c)]))
    print(f"[自校验] s=0,p>0 非 -inf 条目={bad0}（应 0）；s>=1 的 -inf 条目={ninf_hi}（应 0）")

    # 2) 贪心 walk vs Viterbi
    prev, gp, gt = 0, [], 0.0
    for s in range(steps):
        bc, bv = 0, NEG
        for c in range(K):
            v = scores[sc(s, prev, c)]
            if v > bv:
                bv, bc = v, c
        gp.append(bc)
        gt += bv
        prev = bc

    V = [scores[sc(0, 0, c)] for c in range(K)]
    back = []
    for s in range(1, steps):
        nV, nb = [NEG] * K, [0] * K
        for c in range(K):
            bp, bv = 0, NEG
            for p in range(K):
                v = V[p] + scores[sc(s, p, c)]
                if v > bv:
                    bv, bp = v, p
            nV[c], nb[c] = bv, bp
        V = nV
        back.append(nb)
    end_c = max(range(K), key=lambda c: V[c])
    vp = [end_c]
    for s in range(steps - 1, 0, -1):
        vp.append(back[s - 1][vp[-1]])
    vp.reverse()
    vt = V[end_c]

    print(f"[贪心 walk] rank 路径={gp}  路径分={gt:.4f}")
    print(f"[Viterbi  ] rank 路径={vp}  路径分={vt:.4f}")
    d = vt - gt
    print(f"[树材料/打分意义] Viterbi − 贪心 = {d:+.4f}"
          f"  ⇒ {'存在可用材料' if d > 0.5 else '几乎没有（≈0）'}")
    print(f"[贪心 token] {[cand[cd(s, gp[s])] for s in range(steps)]}")
    print(f"[Viterbi token] {[cand[cd(s, vp[s])] for s in range(steps)]}")

    # 3) hit@b（unary 序）
    if a.log:
        lg = pathlib.Path(a.log).read_text(errors="replace")
        cols = {}
        for m in re.finditer(r"\[df2dbg\] col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", lg):
            cols.setdefault(int(m.group(1)), []).append(
                (int(m.group(3)), int(m.group(4)), int(m.group(5))))
        print(f"[dbg] {sum(len(v) for v in cols.values())} 条列记录，最大列={max(cols) if cols else -1}")
        ranks = []
        for s in range(steps):
            if s not in cols:
                ranks.append(99)
                continue
            verify, draft, argmax = cols[s][0]
            cl = [cand[cd(s, c)] for c in range(K)]
            # 只统计该列确实被验证到的轮次（verify 与候选口径不同，这里用 argmax 作 target）
            ranks.append(cl.index(argmax) if argmax in cl else 99)
        print("  逐列 target(argmax) 在候选中的 unary 排名（99=不在候选）:", ranks)
        for b in (1, 2, 4, 8, 16):
            ok = sum(1 for r in ranks if r < b)
            print(f"    hit@{b:<2} = {ok}/{steps} = {100.0*ok/max(steps,1):.1f}%")


if __name__ == "__main__":
    main()
