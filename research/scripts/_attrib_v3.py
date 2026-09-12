import pathlib, re, collections

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
CAND  = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.strip().startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

plain = plain_ids(J / "fx_plain.log")
rounds, cur = [], None
for line in (J / "fx_spec.log").read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        if cur: rounds.append(cur)
        cur = {"accepted": int(m.group(7)), "cols": [], "sel": []}; continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)), "verify": int(m.group(3)),
                            "draft": int(m.group(4)), "argmax": int(m.group(5))}); continue
    m = CAND.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "chosen": int(m.group(3)),
                           "cands": [int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5))]})
if cur: rounds.append(cur)

# 锚定 absolute 位置（col0.verify 在 plain 里前向匹配）
cursor = 0
for r in rounds:
    r["s"] = None
    if r["cols"]:
        head = r["cols"][0]["verify"]
        for i in range(cursor, len(plain)):
            if plain[i] == head: r["s"] = i; cursor = i; break

# 内容配对（候选滞后一轮 + 以 cands[chosen] == 列 draft 配对）
stat = collections.Counter(); pairs = 0
for i, r in enumerate(rounds):
    j = i + 1
    if not (0 <= j < len(rounds)): continue
    nxt = rounds[j]
    if not nxt["cols"] or not r["sel"] or nxt["s"] is None: continue
    by_draft = {}
    for c in nxt["cols"]: by_draft.setdefault(c["draft"], []).append(c)
    own = max(0, nxt["accepted"] if j + 1 >= len(rounds) else rounds[j + 1]["accepted"])
    for st in r["sel"]:
        if not st["cands"] or not (0 <= st["chosen"] < len(st["cands"])): continue
        tok = st["cands"][st["chosen"]]
        cols = by_draft.get(tok, [])
        if not cols: continue
        c = cols[0]
        if c["col"] == 0 or c["col"] > own: continue        # 只统计干净列
        pairs += 1
        key = c["argmax"]                                   # 以 verify 的 argmax 为答案键
        if key == -1: stat["key_dirty"] += 1; continue
        cands = st["cands"]
        if key not in cands: stat["head_miss(verify键)"] += 1
        elif tok != key: stat["walk_error(verify键)"] += 1
        else: stat["accepted"] += 1
        # 顺带：同一格上 plain 与 verify 是否一致
        t = nxt["s"] + c["col"] + 1
        if t < len(plain):
            stat["verify键==plain" if key == plain[t] else "verify键!=plain"] += 1

tot = pairs
print("干净列样本 =", tot)
print("\n=== 以 verify 的 argmax 为答案键的归因（分离“候选质量”与“verify≠plain”）===")
for k, v in stat.most_common():
    print("  %-24s %4d" % (k, v))
if tot:
    hm = stat["head_miss(verify键)"]; we = stat["walk_error(verify键)"]; ac = stat["accepted"]
    print("\n  head_miss = %d/%d = %.1f%%   walk_error = %d/%d = %.1f%%   accepted = %d" %
          (hm, tot, 100.0*hm/tot, we, tot, 100.0*we/tot, ac))
