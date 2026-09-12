#!/usr/bin/env python3
"""草稿质量退化分析：col0 的 draft/argmax/verify 分布 + 命中率 + 回声假设检验。"""
import pathlib, re, collections
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

def parse(p):
    rounds, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            cur = {"anchor": int(m.group(2)), "frontier": int(m.group(3)),
                   "accepted": int(m.group(7)), "cols": []}
            rounds.append(cur); continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return rounds

for tag in ("w8", "w2"):
    rd = parse(D / f"rd_{tag}.log")
    c0 = [r["cols"][0] for r in rd if r["cols"]]
    print("="*70)
    print(f"[{tag}] 轮数={len(rd)}")
    hit = sum(1 for c in c0 if c["draft"] == c["argmax"])
    print(f"  col0 草稿命中率 (draft==argmax) = {hit}/{len(c0)} = {100.0*hit/len(c0):.1f}%")
    print("  col0 draft  出现最多的 8 个:")
    for v, n in collections.Counter(c["draft"] for c in c0).most_common(8):
        print(f"     {v:>8}  x{n}")
    print("  col0 argmax 出现最多的 8 个:")
    for v, n in collections.Counter(c["argmax"] for c in c0).most_common(8):
        print(f"     {v:>8}  x{n}")
    print("  col0 verify 出现最多的 5 个:")
    for v, n in collections.Counter(c["verify"] for c in c0).most_common(5):
        print(f"     {v:>8}  x{n}")
    # 回声假设：本轮 col0 draft 是否等于上一轮某列的 argmax
    echo_prev0 = sum(1 for i in range(1, len(rd)) if rd[i]["cols"] and rd[i-1]["cols"]
                     and rd[i]["cols"][0]["draft"] == rd[i-1]["cols"][0]["argmax"])
    echo_prevA = sum(1 for i in range(1, len(rd)) if rd[i]["cols"] and rd[i-1]["cols"]
                     and rd[i]["cols"][0]["draft"] in [c["argmax"] for c in rd[i-1]["cols"]])
    echo_selfv = sum(1 for c in c0 if c["draft"] == c["verify"])
    print(f"  回声检验: draft0==上轮argmax0  {echo_prev0}/{max(1,len(rd)-1)}"
          f" ; draft0∈上轮任何argmax {echo_prevA}/{max(1,len(rd)-1)}"
          f" ; draft0==本轮verify {echo_selfv}/{len(c0)}")
    # 全列草稿命中率（分列）
    W = max(len(r["cols"]) for r in rd if r["cols"])
    for j in range(W):
        cs = [r["cols"][j] for r in rd if len(r["cols"]) > j]
        h = sum(1 for c in cs if c["draft"] == c["argmax"] and c["draft"] != -1)
        v = sum(1 for c in cs if c["draft"] != -1)
        print(f"    列{j}: 有草稿 {v:>3} 命中 {h:>3} = {100.0*h/v if v else 0:>5.1f}%")
