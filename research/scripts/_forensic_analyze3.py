import json, struct, pathlib, re, collections, numpy as np

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")
with open(ART, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
o = objs["text/draft_head_token_ids"]
with open(ART, "rb") as f:
    f.seek(payload + o["offset"]); table = np.frombuffer(f.read(o["bytes"]), dtype=np.int32)
inv = {int(v): i for i, v in enumerate(table)}     # vocab id -> draft row（备用）
def d2v(i): return int(table[i]) if 0 <= i < len(table) else -1

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
        cur = {"cols": [], "sel": [], "accepted": int(m.group(7))}
        continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)), "draft": int(m.group(4)),
                            "argmax": int(m.group(5))}); continue
    m = CAND.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "chosen": int(m.group(3)),
                           "cands": [int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5))]})
if cur: rounds.append(cur)

# 探针的 accepted 滞后一轮 ⇒ 本轮 accepted = 下一轮的 accepted 字段
for i, r in enumerate(rounds):
    r["own"] = rounds[i+1]["accepted"] if i+1 < len(rounds) else None

stat = collections.Counter(); by_step = collections.defaultdict(collections.Counter)
stat_raw = collections.Counter()
samples = []
for r in rounds:
    if not r["cols"] or not r["sel"] or r["own"] is None: continue
    p0 = r["cols"][0]["pos"]
    own = max(0, r["own"])
    for st in r["sel"]:
        s = st["s"]
        if s >= own:                     # ← 只统计落在"已接受前缀"内的步（干净列）
            continue
        tgt = p0 + 1 + s
        if tgt >= len(plain): continue
        correct = plain[tgt]
        cands = st["cands"]
        cands_v = [d2v(c) for c in cands]
        chosen_v = cands_v[st["chosen"]] if 0 <= st["chosen"] < len(cands_v) else None
        in_c = correct in cands_v
        rk = cands_v.index(correct) if in_c else -1
        if not in_c: cls = "head_miss"
        elif chosen_v != correct: cls = "walk_error"
        else:
            col = next((c for c in r["cols"] if c["col"] == s + 1), None)
            cls = "accepted" if (col and col["argmax"] == correct) else "verify_flip"
        stat[cls] += 1; by_step[s][cls] += 1
        if len(samples) < 8 and cls in ("head_miss", "walk_error"):
            samples.append((s, tgt, correct, chosen_v, rk, cls, cands_v))

tot = sum(stat.values())
print("=== 修正后（仅干净列）===  样本 = %d 步" % tot)
for k, v in stat.most_common():
    print("  %-12s %5d  %5.1f%%" % (k, v, 100.0*v/tot))
print("\n  s | 步数 | head_miss | walk_error | verify_flip | accepted")
for s in sorted(by_step):
    c = by_step[s]; n = sum(c.values())
    print("  %d | %4d | %9d | %9d | %10d | %8d" % (s, n, c["head_miss"], c["walk_error"], c["verify_flip"], c["accepted"]))
print("\n=== 样例 ===")
for s, tp, cor, ch, rk, cls, cs in samples:
    print("  s=%d pos=%d 正确=%d walk选=%s rank=%d %s cs=%s" % (s, tp, cor, ch, rk, cls, cs[:6]))
