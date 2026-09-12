#!/usr/bin/env python3
"""hit@b（唯一对齐版）：key = (dump.anchors[0], dump.frontiers[0]) ↔ (dbg 轮 col0.verify, 轮前 frontier)。

dbg 的 frontier/anchor 是**轮后**值（它们同时是 accept kernel 的输出），故轮前 frontier = dbg.frontier - count。
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
    lg = pathlib.Path(DL / "s2_scores.log").read_text(errors="replace")
    rounds = []
    cur = None
    for line in lg.splitlines():
        m = re.search(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                      r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)", line)
        if m:
            cur = {"row": int(m.group(1)), "anchor": int(m.group(2)), "frontier": int(m.group(3)),
                   "accepted": int(m.group(7)), "count": int(m.group(8)), "cols": {}}
            rounds.append(cur)
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            cur["cols"][int(m.group(1))] = {"pos": int(m.group(2)), "verify": int(m.group(3)),
                                            "draft": int(m.group(4)), "argmax": int(m.group(5))}
    print(f"dbg 轮数={len(rounds)}")
    for r in rounds:
        r["pre"] = r["frontier"] - r["count"]

    hits = {}
    matched = 0
    print("\ncall | anchor | frontier | 对齐到 dbg 轮 | 逐深度 target 在(分数序)候选中的排名")
    for n in range(1, 40):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            break
        scores = load(fs, "f")
        cand = load(DL / f"df2scores_cand_{n}.bin", "i")
        front = load(DL / f"df2scores_front_{n}.bin", "i")
        anch = load(DL / f"df2scores_anch_{n}.bin", "i")
        steps = len(scores) // (K * K)

        def sc(s, p, c):
            return s + steps * (p + K * c)

        # 该 call 的贪心路径
        prev, path = 0, []
        for s in range(steps):
            bv, bc = float("-inf"), 0
            for c in range(K):
                v = scores[sc(s, prev, c)]
                if v > bv:
                    bv, bc = v, c
            path.append(bc)
            prev = bc

        a0, f0 = anch[0], front[0]
        m_r = [r for r in rounds if r["cols"].get(0, {}).get("verify") == a0 and r["pre"] == f0]
        if len(m_r) != 1:
            m_r2 = [r for r in rounds if r["cols"].get(0, {}).get("verify") == a0]
            tag = f"锚点{len(m_r2)}个候选" + ("" if not m_r2 else f"（frontier {[x['pre'] for x in m_r2]} vs dump {f0}）")
            print(f"{n:>4} | {a0:>6} | {f0:>8} | 未唯一对齐：{tag}")
            continue
        r = m_r[0]
        matched += 1
        ranks = []
        prev = 0
        for s in range(steps):
            if s not in r["cols"]:
                ranks.append(None); continue
            tgt = r["cols"][s]["argmax"]
            order = sorted(range(K), key=lambda c: -scores[sc(s, prev, c)])
            toks = [cand[s + steps * c] for c in order]
            ranks.append(toks.index(tgt) if tgt in toks else 99)
            bv, bc = float("-inf"), 0
            for c in range(K):
                v = scores[sc(s, prev, c)]
                if v > bv:
                    bv, bc = v, c
            prev = bc
        print(f"{n:>4} | {a0:>6} | {f0:>8} | 轮{r['row']:>3} (accepted={r['accepted']}) | {ranks}")
        for s, rk in enumerate(ranks):
            if rk is not None:
                hits.setdefault(s, []).append(rk)

    print(f"\n成功唯对齐的 call = {matched}")
    print("=== 逐深度 hit@b（唯一对齐样本）===")
    print(" 深度 | 样本 | hit@1 | hit@2 | hit@4 | hit@8 | hit@16")
    for s in sorted(hits):
        rs = hits[s]
        nn = len(rs)
        row = " ".join(f"{100.0*sum(1 for x in rs if x < b)/nn:5.1f}%" for b in (1, 2, 4, 8, 16))
        print(f" {s:>4} | {nn:>4} | {row}")


if __name__ == "__main__":
    main()
