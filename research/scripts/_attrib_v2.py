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

stat = collections.Counter(); by_s = collections.defaultdict(collections.Counter)
samples = []
paired = unmatched = 0
for i, r in enumerate(rounds):
    j = i + 1                       # 已验证：候选滞后一轮
    if not (0 <= j < len(rounds)): continue
    nxt = rounds[j]
    if not nxt["cols"] or not r["sel"]: continue
    by_draft = {}
    for c in nxt["cols"]:
        by_draft.setdefault(c["draft"], []).append(c)
    # 该轮的接受数：用"它自己那一轮"的 accepted（探针滞后一拍，故取 nxt 的 accepted）
    own = max(0, nxt["accepted"] if j + 1 >= len(rounds) else rounds[j + 1]["accepted"])
    for st in r["sel"]:
        if not st["cands"]: continue
        chosen_tok = st["cands"][st["chosen"]] if 0 <= st["chosen"] < len(st["cands"]) else None
        cols = by_draft.get(chosen_tok, [])
        if not cols:
            unmatched += 1; continue
        col = cols[0]
        paired += 1
        if col["col"] > own or col["col"] == 0:
            continue                # 只统计落在已接受前缀内的干净列（不含 col0：那是 anchor 位）
        tgt = col["pos"] + 1
        if tgt >= len(plain): continue
        correct = plain[tgt]
        cands = st["cands"]
        in_c = correct in cands
        if not in_c: cls = "head_miss"
        elif chosen_tok != correct: cls = "walk_error"
        else:
            cls = "accepted" if col["argmax"] == correct else "verify_flip"
        stat[cls] += 1; by_s[st["s"]][cls] += 1
        if len(samples) < 10 and cls in ("head_miss", "walk_error"):
            samples.append((st["s"], col["col"], tgt, correct, chosen_tok, cls, cands[:6]))

print("内容配对成功 = %d 步，未配上 = %d" % (paired, unmatched))
tot = sum(stat.values())
print("\n=== 可验证对齐下的归因（仅干净列，共 %d 步）===" % tot)
for k, v in stat.most_common():
    print("  %-12s %5d  %5.1f%%" % (k, v, 100.0 * v / tot if tot else 0))
print("\n  s | 步数 | head_miss | walk_error | verify_flip | accepted")
for s in sorted(by_s):
    c = by_s[s]; n = sum(c.values())
    print("  %d | %4d | %9d | %9d | %10d | %8d" %
          (s, n, c["head_miss"], c["walk_error"], c["verify_flip"], c["accepted"]))
print("\n=== 样例 ===")
for s, cc, tp, cor, ch, cls, cs in samples:
    print("  s=%d col=%d pos+1=%d 正确=%d 选中=%d %s cs=%s" % (s, cc, tp, cor, ch, cls, cs))
