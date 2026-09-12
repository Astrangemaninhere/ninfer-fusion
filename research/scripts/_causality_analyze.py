#!/usr/bin/env python3
"""因果性判定（纯分析）：读两份 DF2DBG 探针日志，按"相同上下文前缀 + 相同位置"配对，
比较第 0 列 argmax。控制项：两 run 的右侧 draft 列应当不同（否则实验无力度）。
"""
import pathlib
import re
import sys

ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")


def parse(path):
    rounds, cur = [], None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            cur = {"accepted": int(m.group(7)), "cols": []}
            rounds.append(cur)
            continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return rounds


def trajectory(rounds):
    out, prefix = [], []
    for i, r in enumerate(rounds):
        own = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
        if own is None or not r["cols"]:
            continue
        a = max(0, own)
        drafts = [c["draft"] for c in r["cols"] if c["draft"] != -1]
        out.append({"prefix": tuple(prefix), "pos": r["cols"][0]["pos"],
                    "anchor": r["cols"][0]["verify"], "col0": r["cols"][0]["argmax"],
                    "rest": tuple(drafts[1:]) if len(drafts) > 1 else ()})
        prefix.extend(drafts[:a])
        prefix.append(r["cols"][min(a, len(r["cols"]) - 1)]["argmax"])
    return out


def main() -> int:
    a = trajectory(parse("/tmp/caus_full2.log"))
    b = trajectory(parse("/tmp/caus_lmhd.log"))
    print(f"轮数: full={len(a)} lmhd={len(b)}")
    index = {}
    for r in b:
        index.setdefault((r["prefix"], r["pos"]), []).append(r)
    pairs = [(r, q) for r in a for q in index.get((r["prefix"], r["pos"]), [])]
    print(f"可配对轮（同上下文前缀 + 同位置）: {len(pairs)}")
    if not pairs:
        # 放宽：只按位置 + anchor 配对（上下文尾部相同即可）
        idx2 = {}
        for r in b:
            idx2.setdefault((r["anchor"], r["pos"]), []).append(r)
        pairs = [(r, q) for r in a for q in idx2.get((r["anchor"], r["pos"]), [])]
        print(f"  （放宽为 anchor+pos 配对）: {len(pairs)}")
    if not pairs:
        print("=> 仍无配对，需更短 prompt / 更少接受差异")
        return 0
    diff = [p for p in pairs if p[0]["col0"] != p[1]["col0"]]
    rest_diff = sum(1 for p in pairs if p[0]["rest"] != p[1]["rest"])
    print()
    print(f"  col0 argmax 不同: {len(diff)}/{len(pairs)} = {100.0*len(diff)/len(pairs):.1f}%")
    print(f"  右侧 draft 列不同: {rest_diff}/{len(pairs)}  ← 控制项（应接近 100%）")
    print()
    for r, q in pairs[:6]:
        tag = "  <-- col0 不同" if r["col0"] != q["col0"] else ""
        print(f"  pos={r['pos']:<5} anchor={r['anchor']:<8} prefix_len={len(r['prefix']):<4} "
              f"col0: A={r['col0']:<8} B={q['col0']:<8} restA={list(r['rest'][:3])} "
              f"restB={list(q['rest'][:3])}{tag}")
    print()
    if diff:
        print("⇒ col0 会随右侧草稿变化 ⇒ **target verify 非因果（掩码 bug）**")
    else:
        print("⇒ col0 与右侧草稿无关 ⇒ 因果性成立（核心掩码不是本因）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
