#!/usr/bin/env python3
"""树上界（修正解析版）：整轮解析完再按 col0.pos 建索引。

前版 bug：在解析到 col=0 那一行时就 dict(cur) 快照，导致每轮只剩 {0: ...}，
于是树的高宽全等于链。本版修好后重算。
"""
import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16


def load(p, f):
    raw = p.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}{f}", raw))


def main():
    lg = (DL / "s3_scores.log").read_text(errors="replace")
    rounds, cur = [], None
    for line in lg.splitlines():
        if re.search(r"\[df2dbg\] row=(\d+)", line):
            cur = {}
            rounds.append(cur)
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            cur[int(m.group(1))] = {"pos": int(m.group(2)), "verify": int(m.group(3)),
                                    "draft": int(m.group(4)), "argmax": int(m.group(5))}
    pos_index = {}
    for r in rounds:
        if 0 in r:
            pos_index[r[0]["pos"]] = r
    print(f"轮数={len(rounds)}  可按 col0.pos 索引的轮={len(pos_index)}  "
          f"每轮列数中位={sorted(len(r) for r in rounds)[len(rounds)//2]}")

    def sc(s, steps, p, c):
        return s + steps * (p + K * c)

    chains, t2, t4, t16, ds = [], [], [], [], []
    for n in range(1, 40):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            break
        scores = load(fs, "f")
        cand = load(DL / f"df2scores_cand_{n}.bin", "i")
        front = load(DL / f"df2scores_front_{n}.bin", "i")
        steps = len(scores) // (K * K)
        r = pos_index.get(front[0])
        if r is None:
            print(f"  call {n}: frontier={front[0]} 无对应轮，跳过")
            continue
        tgt = {c: r[c]["argmax"] for c in r}

        def gp_path():
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

        gp = gp_path()
        gsc = sum(scores[sc(s, steps, gp[s - 1] if s else 0, gp[s])] for s in range(steps))
        V = [scores[sc(0, steps, 0, c)] for c in range(K)]
        for s in range(1, steps):
            V = [max(V[p] + scores[sc(s, steps, p, c)] for p in range(K)) for c in range(K)]
        ds.append(max(V) - gsc)

        # 链：按贪心路径逐位判定
        acc = 0
        prev = 0
        for s in range(steps):
            c0 = sorted(range(K), key=lambda c: -scores[sc(s, steps, prev, c)])[0]
            if s in tgt and cand[s + steps * c0] == tgt[s]:
                acc += 1
                prev = c0
            else:
                break
        chains.append(acc)

        def ev(b):
            c0 = sorted(range(K), key=lambda c: -scores[sc(0, steps, 0, c)])[0]
            if 0 not in tgt or cand[0 + steps * c0] != tgt[0]:
                return 0
            a, fr = 1, {c0}
            for s in range(1, steps):
                if s not in tgt:
                    break
                nxt = set()
                for p in fr:
                    for c in sorted(range(K), key=lambda c: -scores[sc(s, steps, p, c)])[:b]:
                        if cand[s + steps * c] == tgt[s]:
                            nxt.add(c)
                if not nxt:
                    break
                a = s + 1
                fr = nxt
            return a

        t2.append(ev(2)); t4.append(ev(4)); t16.append(ev(K))

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    print(f"\n样本={len(chains)}")
    print(f"EAL(链)     = 1 + {avg(chains):.2f} = {1+avg(chains):.2f}   （引擎实测 AL=1.32）")
    print(f"EAL(树 b=2) = 1 + {avg(t2):.2f} = {1+avg(t2):.2f}")
    print(f"EAL(树 b=4) = 1 + {avg(t4):.2f} = {1+avg(t4):.2f}")
    print(f"EAL(全集16) = 1 + {avg(t16):.2f} = {1+avg(t16):.2f}  ← 候选集天花板")
    print(f"Viterbi−贪心 路径分 = {avg(ds):+.2f}")
    print("\n逐样本 (call, 链, b2, b4, 16, Δ分):")
    for i, (c, a, b, d, e) in enumerate(zip(chains, t2, t4, t16, ds)):
        print(f"  {i+1:>2}: {c} {a} {b} {d}  {e:+.2f}")


if __name__ == "__main__":
    main()
