import re, pathlib, collections

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.strip().startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
SEL   = re.compile(r"\[df2sel\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) tok=(-?\d+) E=([-\d.]+) u=([-\d.]+) "
                   r"pair=([-\d.]+) uspan=([-\d.]+) pairspan=([-\d.]+) Earg=(-?\d+) Uarg=(-?\d+)")
CAND  = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")

plain = plain_ids(J / "fx_plain.log")
print("plain ids =", len(plain))

# 逐轮解析：anchor 位置 = 该轮 col0 的 pos
rounds = []
cur = None
pending_sel = []   # 本轮内按顺序累积的 [df2sel] / [df2cand]
for line in (J / "fx_spec.log").read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        if cur: rounds.append(cur)
        cur = {"anchor": int(m.group(2)), "accepted": int(m.group(7)), "cols": [], "sel": []}
        continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                            "verify": int(m.group(3)), "draft": int(m.group(4)), "argmax": int(m.group(5))})
        continue
    m = CAND.search(line)
    if m:
        pairs = re.findall(r"(-?\d+):([-\d.]+)", m.group(5))
        cur["sel"].append({"s": int(m.group(1)), "pred": int(m.group(2)), "chosen": int(m.group(3)),
                           "k": int(m.group(4)), "cands": [int(a) for a, _ in pairs],
                           "unary": [float(b) for _, b in pairs]})
        continue
    m = SEL.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "Earg": int(m.group(10)), "Uarg": int(m.group(11))})
if cur: rounds.append(cur)

print("轮数 =", len(rounds), " 带候选的轮 =", sum(1 for r in rounds if r["sel"]))

stat = collections.Counter()
by_step = collections.defaultdict(collections.Counter)
rows = []
for r in rounds:
    if not r["cols"] or not r["sel"]: continue
    p0 = r["cols"][0]["pos"]                      # anchor 位置
    for st in r["sel"]:
        s = st["s"]
        if "cands" not in st: continue
        tgt_pos = p0 + 1 + s                      # 这一步的提案要预测的位置
        if tgt_pos >= len(plain): continue
        correct = plain[tgt_pos]
        cands = st["cands"]
        chosen_tok = cands[st["chosen"]] if 0 <= st["chosen"] < len(cands) else None
        in_cands = correct in cands
        rank = cands.index(correct) if in_cands else -1
        if not in_cands:
            cls = "head_miss"                     # 候选集里没有正确答案
        elif chosen_tok != correct:
            cls = "walk_error"                    # 有，但 walk 选了别的
        else:
            # walk 选对：看 verify 是否认（argmax 是否等于 correct）
            col = next((c for c in r["cols"] if c["col"] == s + 1), None)
            if col is None:
                cls = "walk_ok_nocolline"
            elif col["argmax"] == correct:
                cls = "accepted"
            elif col["argmax"] == -1:
                cls = "walk_ok_dirty"
            else:
                cls = "verify_flip"               # 选对了，但 verify 的 argmax 是别的
        stat[cls] += 1
        by_step[s][cls] += 1
        rows.append((s, tgt_pos, correct, chosen_tok, rank, cls, st))

tot = sum(stat.values())
print("\n=== 逐步归因（共 %d 步） ===" % tot)
for k, v in stat.most_common():
    print("  %-18s %5d  %5.1f%%" % (k, v, 100.0*v/tot))

print("\n=== 按 walk 步序 ===")
print("  s | 步数 | head_miss | walk_error | verify_flip | accepted")
for s in sorted(by_step):
    c = by_step[s]; n = sum(c.values())
    print("  %d | %4d | %9d | %9d | %10d | %8d" %
          (s, n, c["head_miss"], c["walk_error"], c["verify_flip"], c["accepted"]))

print("\n=== 前 15 个 walk_error / head_miss 样例（s, 目标位置, 正确, walk选, 正确在候选里的 rank） ===")
for s, tp, cor, ch, rk, cls, st in rows:
    if cls in ("walk_error", "head_miss"):
        print("  s=%d pos=%d correct=%d chosen=%d rank=%d %s  cs=%s" %
              (s, tp, cor, ch if ch is not None else -1, rk, cls, st["cands"][:6]))
