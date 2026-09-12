import pathlib, re, collections

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

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
        cur = {"accepted": int(m.group(7)), "cols": []}; continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)), "verify": int(m.group(3)),
                            "argmax": int(m.group(5))})
if cur: rounds.append(cur)

# 锚定：把每轮 col0 的 verify 在 plain 序列里前向匹配（确定绝对位置）
cursor = 0
for r in rounds:
    r["s"] = None
    if r["cols"]:
        head = r["cols"][0]["verify"]
        for i in range(cursor, len(plain)):
            if plain[i] == head: r["s"] = i; cursor = i; break

# 逐列：verify 的输入 token 是否与 plain 序列一致（列 j 的 verify 应等于 plain[s+j]）
stat = collections.Counter()
for r in rounds:
    if r["s"] is None: continue
    for c in r["cols"]:
        t = r["s"] + c["col"]
        if t >= len(plain): continue
        if c["verify"] == plain[t]:
            stat["verify输入==plain[同位置]"] += 1
        else:
            stat["verify输入!=plain[同位置]"] += 1

print("=== 逐列一致性（verify 吃到的 token 流 vs plain 输出）===")
for k, v in stat.most_common():
    print("  %-32s %d" % (k, v))

# 干净列上：argmax 是否等于 plain 的下一 token（=验证"answer key"是否正确）
acc = collections.Counter()
for r in rounds:
    if r["s"] is None: continue
    own = max(0, r["accepted"])
    for c in r["cols"]:
        j = c["col"]
        if j == 0 or j > own: continue
        t = r["s"] + j + 1
        if t >= len(plain) or c["argmax"] == -1: continue
        acc["argmax==plain[pos+1]" if c["argmax"] == plain[t] else "argmax!=plain[pos+1]"] += 1
print("\n=== 干净列上 verify 的 argmax 与 plain 下一 token 的关系（决定 answer key 是否成立）===")
for k, v in acc.most_common():
    print("  %-28s %d" % (k, v))
