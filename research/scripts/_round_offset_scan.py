import pathlib, re

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
CAND  = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")

rounds, cur = [], None
for line in (J / "fx_spec.log").read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        if cur: rounds.append(cur)
        cur = {"cols": [], "sel": []}; continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "draft": int(m.group(4))}); continue
    m = CAND.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "chosen": int(m.group(3)),
                           "cands": [int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5))]})
if cur: rounds.append(cur)

print("轮数 =", len(rounds))
print("每轮 (选中数, 列数):", [(len(r["sel"]), len(r["cols"])) for r in rounds[:8]])

dead = [r for r in rounds if not r["cols"] or not r["sel"]]
print("有列或选中为空的轮 =", len(dead))

# 偏移扫描：把第 i 轮的 sel 与第 i+off 轮的 cols 比对（同轮内再按 s+1 / s 对齐）
print("\n=== 轮次偏移扫描（匹配率 = cands[chosen] 能在目标轮某列 draft 里找到的比例）===")
for off in (-2, -1, 0, 1, 2):
    hit = tot = 0
    for i, r in enumerate(rounds):
        j = i + off
        if not (0 <= j < len(rounds)): continue
        drafts = {c["draft"] for c in rounds[j]["cols"]}
        if not drafts: continue
        for st in r["sel"]:
            if not st["cands"]: continue
            c = st["cands"][st["chosen"]] if 0 <= st["chosen"] < len(st["cands"]) else None
            if c is None: continue
            tot += 1; hit += (c in drafts)
    print("  off=%+d : %d/%d = %.1f%%" % (off, hit, tot, 100.0*hit/tot if tot else 0))
