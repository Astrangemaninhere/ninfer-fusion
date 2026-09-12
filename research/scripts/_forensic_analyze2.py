import json, struct, pathlib, re, collections, numpy as np

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")

# 1) 读出草稿域映射表
with open(ART, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
o = objs["text/draft_head_token_ids"]
with open(ART, "rb") as f:
    f.seek(payload + o["offset"])
    table = np.frombuffer(f.read(o["bytes"]), dtype=np.int32)
print("映射表: %d 项, 范围 [%d,%d], 前10=%s" % (len(table), table.min(), table.max(), table[:10].tolist()))
def d2v(i):
    return int(table[i]) if 0 <= i < len(table) else -1

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.strip().startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
CAND  = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")

plain = plain_ids(J / "fx_plain.log")
rounds, cur = [], None
for line in (J / "fx_spec.log").read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        if cur: rounds.append(cur)
        cur = {"cols": [], "sel": []}
        continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)), "draft": int(m.group(4)),
                            "argmax": int(m.group(5))})
        continue
    m = CAND.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "chosen": int(m.group(3)),
                           "cands": [int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5))]})
if cur: rounds.append(cur)
print("轮数 =", len(rounds))

stat = collections.Counter(); by_step = collections.defaultdict(collections.Counter)
rows = []
for r in rounds:
    if not r["cols"] or not r["sel"]: continue
    p0 = r["cols"][0]["pos"]
    for st in r["sel"]:
        s = st["s"]; tgt = p0 + 1 + s
        if tgt >= len(plain): continue
        correct = plain[tgt]
        cands_v = [d2v(c) for c in st["cands"]]          # ← 关键：映射回真词表
        chosen_v = cands_v[st["chosen"]] if 0 <= st["chosen"] < len(cands_v) else None
        in_c = correct in cands_v
        rk = cands_v.index(correct) if in_c else -1
        if not in_c: cls = "head_miss"
        elif chosen_v != correct: cls = "walk_error"
        else:
            col = next((c for c in r["cols"] if c["col"] == s + 1), None)
            cls = "accepted" if (col and col["argmax"] == correct) else ("verify_flip" if col else "no_col")
        stat[cls] += 1; by_step[s][cls] += 1
        rows.append((s, tgt, correct, chosen_v, rk, cls, cands_v))

tot = sum(stat.values())
print("\n=== 修正后的逐步归因（共 %d 步，候选已映射回真词表） ===" % tot)
for k, v in stat.most_common():
    print("  %-12s %5d  %5.1f%%" % (k, v, 100.0*v/tot))
print("\n  s | 步数 | head_miss | walk_error | verify_flip | accepted")
for s in sorted(by_step):
    c = by_step[s]; n = sum(c.values())
    print("  %d | %4d | %9d | %9d | %10d | %8d" % (s, n, c["head_miss"], c["walk_error"], c["verify_flip"], c["accepted"]))
print("\n=== 样例（前 12 条 head_miss / walk_error）===")
for s, tp, cor, ch, rk, cls, cs in rows:
    if cls in ("head_miss", "walk_error"):
        print("  s=%d pos=%d 正确=%d walk选=%s rank=%d %s cs=%s" % (s, tp, cor, ch, rk, cls, cs[:6]))
