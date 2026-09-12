#!/usr/bin/env python3
"""hit@b：把 dump 的候选(按 selector 分数排序)与 [df2dbg] 的 target argmax 逐轮对齐，
回答"若在深度 s 试 b 个节点，target 的真 token 有多少比例落在这 b 个里"。

对齐键：dump 的 s=0 贪心 token == 该轮 [df2dbg] 的 draft(col=0)。
"""
import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16


def load(path, fmt):
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}{fmt}", raw))


def dump_call(n):
    scores = load(DL / f"df2scores_scores_{n}.bin", "f")
    cand = load(DL / f"df2scores_cand_{n}.bin", "i")
    steps = len(scores) // (K * K)
    return steps, scores, cand


def main():
    lg = pathlib.Path(DL / "sv_scores.log").read_text(errors="replace")
    rounds = []
    cur = None
    for line in lg.splitlines():
        m = re.search(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                      r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)", line)
        if m:
            cur = {"row": int(m.group(1)), "cols": {}}
            rounds.append(cur)
            continue
        m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
        if m and cur is not None:
            cur["cols"][int(m.group(1))] = {"pos": int(m.group(2)), "verify": int(m.group(3)),
                                            "draft": int(m.group(4)), "argmax": int(m.group(5))}
    print(f"dbg 轮数={len(rounds)}")
    ok = [r for r in rounds if 0 in r["cols"]]
    print(f"含 col=0 的轮={len(ok)}")

    # 建立 dump 索引：call -> (steps, scores, cand)，并算其 s=0 贪心 token
    dumps = {}
    for n in range(1, 13):
        p = DL / f"df2scores_scores_{n}.bin"
        if not p.exists():
            continue
        steps, scores, cand = dump_call(n)
        def sc(s, pp, c):
            return s + steps * (pp + K * c)
        prev, gp = 0, []
        for s in range(steps):
            bc, bv = 0, float("-inf")
            for c in range(K):
                v = scores[sc(s, prev, c)]
                if v > bv:
                    bv, bc = v, c
            gp.append(bc)
            prev = bc
        dumps[n] = (steps, scores, cand, cand[gp[0]] if steps else -1)

    print("\ncall | s=0 贪心 token | 对齐到的轮 | 逐深度 target 在候选中的**分数排名**")
    hits_by_depth = {}
    for n, (steps, scores, cand, tok0) in sorted(dumps.items()):
        match = next((r for r in ok if r["cols"][0]["draft"] == tok0), None)
        if match is None:
            print(f"{n:>4} | {tok0:>13} | 未对齐（warmup 轮不在 dbg 里）")
            continue
        def sc(s, pp, c):
            return s + steps * (pp + K * c)
        ranks = []
        for s in range(steps):
            if s not in match["cols"]:
                ranks.append(None); continue
            tgt = match["cols"][s]["argmax"]
            # 该深度按"分数"排序：用贪心到达的 pred 行排序（这是实际会用到的行）
            prev = 0
            for ss in range(s):
                bv, bc = float("-inf"), 0
                for c in range(K):
                    v = scores[sc(ss, prev, c)]
                    if v > bv:
                        bv, bc = v, c
                prev = bc
            order = sorted(range(K), key=lambda c: -scores[sc(s, prev, c)])
            toks = [cand[s + steps * c] for c in order]
            ranks.append(toks.index(tgt) if tgt in toks else 99)
        print(f"{n:>4} | {tok0:>13} | 轮{len(rounds)-len(ok)+rounds.index(match) if match in rounds else '?'} | {ranks}")
        for s, r in enumerate(ranks):
            if r is not None:
                hits_by_depth.setdefault(s, []).append(r)

    print("\n=== 逐深度 hit@b（仅统计已对齐的轮）===")
    print(" 深度 | 样本 | hit@1 | hit@2 | hit@4 | hit@8 | hit@16")
    for s in sorted(hits_by_depth):
        rs = hits_by_depth[s]
        n = len(rs)
        row = " ".join(f"{100.0*sum(1 for r in rs if r < b)/n:5.1f}%" for b in (1, 2, 4, 8, 16))
        print(f" {s:>4} | {n:>4} | {row}")


if __name__ == "__main__":
    main()
