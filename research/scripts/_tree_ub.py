#!/usr/bin/env python3
"""离线兑现树上界：用 12 组真实 16x16 分数矩阵 + dbg 的 target 真 token，
对比 链式 EAL(实测) vs 树(b, 深度 0 单链/深度≥1 分支) 的**乐观上界**。

⚠️ 乐观性说明：dbg 的 target token 是在**链式前缀**条件下产生的；树会走不同前缀 ⇒ target token 会变
⇒ 本结果是上界，不是实测。所有结论必须标注为"上界"。
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
    rounds, cur = [], None
    for line in lg.splitlines():
        m = re.search(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+)", line)
        if m:
            cur = {"cols": {}}
            rounds.append(cur)
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            cur["cols"][int(m.group(1))] = int(m.group(5))

    rows = []
    for n in range(1, 40):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            break
        scores = load(fs, "f")
        cand = load(DL / f"df2scores_cand_{n}.bin", "i")
        front = load(DL / f"df2scores_front_{n}.bin", "i")
        steps = len(scores) // (K * K)

        def sc(s, p, c):
            return s + steps * (p + K * c)

        # 对齐（唯一键：frontier == dbg col0.pos）
        m_r = [r for r in rounds if r["cols"].get(0) is not None and r.get("pos0") == front[0]]
        # dbg 的 pos 在 col0 记录里
        m_r = []
        for r in rounds:
            if 0 in r["cols"]:
                pass
        # 直接按 frontier 与 col=0 的 pos 对齐（pos 存在列记录里，这里另建索引）
        rows.append((n, steps, scores, cand, front[0]))

    # 重建 dbg 的 pos→round 索引
    pos_index = {}
    cur = None
    for line in lg.splitlines():
        m = re.search(r"\[df2dbg\] row=(\d+)", line)
        if m:
            cur = {"cols": {}}
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            col, pos, argmax = int(m.group(1)), int(m.group(2)), int(m.group(5))
            cur["cols"][col] = argmax
            if col == 0:
                pos_index[pos] = dict(cur["cols"])
                pos_index[pos]["__pos0__"] = pos

    def eval_chain(steps, scores, cand, tgt):
        prev, acc = 0, 0
        for s in range(steps):
            bv, bc = float("-inf"), 0
            for c in range(K):
                v = scores[sc_local(s, prev, c)]
                if v > bv:
                    bv, bc = v, c
            if tgt.get(s) is not None and cand[s + steps * bc] == tgt[s]:
                acc += 1
                prev = bc
            else:
                break
        return acc

    def eval_tree(steps, scores, cand, tgt, b, start=1):
        # 深度 0：单链
        bv, r0 = float("-inf"), 0
        for c in range(K):
            v = scores[sc_local(0, 0, c)]
            if v > bv:
                bv, r0 = v, c
        if tgt.get(0) is None or cand[0 + steps * r0] != tgt[0]:
            return 0
        acc, frontier = 1, {r0}
        for s in range(1, steps):
            if tgt.get(s) is None:
                break
            width = b if s >= start else 1
            nxt = set()
            for p in frontier:
                order = sorted(range(K), key=lambda c: -scores[sc_local(s, p, c)])[:width]
                for c in order:
                    if cand[s + steps * c] == tgt[s]:
                        nxt.add(c)
            if not nxt:
                break
            acc = s + 1
            frontier = nxt
        return acc

    def sc_local(s, p, c):
        return s + steps_cur[0] * (p + K * c)

    chains, trees2, trees4, oracles, ns = [], [], [], [], 0
    for n, steps, scores, cand, front0 in rows:
        steps_cur[0] = steps
        tgt = pos_index.get(front0)
        if tgt is None:
            continue
        ns += 1
        chains.append(eval_chain(steps, scores, cand, tgt))
        trees2.append(eval_tree(steps, scores, cand, tgt, 2))
        trees4.append(eval_tree(steps, scores, cand, tgt, 4))
        oracles.append(eval_tree(steps, scores, cand, tgt, K))

    def stat(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"对齐样本 = {ns}")
    print(f"EAL(链, 上界口径)      = 1 + {stat(chains):.2f} = {1+stat(chains):.2f}   （引擎实测 AL = 1.32）")
    print(f"EAL(树 b=2, 深度0单链) = 1 + {stat(trees2):.2f} = {1+stat(trees2):.2f}")
    print(f"EAL(树 b=4)            = 1 + {stat(trees4):.2f} = {1+stat(trees4):.2f}")
    print(f"EAL(全集 16)           = 1 + {stat(oracles):.2f} = {1+stat(oracles):.2f}  ← 候选集天花板")
    print()
    print("逐样本（链 / b=2 / b=4 / 16）:")
    for i in range(ns):
        print(f"  {i:>2}: {chains[i]} / {trees2[i]} / {trees4[i]} / {oracles[i]}")


steps_cur = [7]
if __name__ == "__main__":
    main()
