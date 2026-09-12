#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 新运行里是否有 head 指示（summary/request 行） ==="
for f in /home/user/vdf2_lmhd.log /home/user/vdf2_full.log; do
  echo "--- $f ---"
  grep -iE 'proposal|lm.head|draft head|optimized' "$f" | head -6
  grep -E '^summary' "$f" | head -14
done
echo
echo "=== 2) 两次探针的草稿是否相同（草稿头是否真的改变了候选） ==="
python3 - <<'PY'
import pathlib, re
ROUND = re.compile(r"\[df2dbg\] row=\d+ anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                   r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
def parse(p):
    rounds, cur = [], None
    try:
        txt = pathlib.Path(p).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return rounds
    for line in txt:
        m = ROUND.search(line)
        if m:
            cur = {"meta": m.groups(), "cols": []}; rounds.append(cur); continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append(m.groups())
    return rounds
old = parse("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_probe_raw.log")
new = parse("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2head_probe.log")
print(f"old rounds={len(old)}  new rounds={len(new)}")
dif = same = 0
for a, b in zip(old, new):
    if [c[3] for c in a["cols"]] == [c[3] for c in b["cols"]]:
        same += 1
    else:
        dif += 1
print(f"轮次中草稿向量相同的: {same}   不同的: {dif}")
print()
for i, (a, b) in enumerate(zip(old, new)):
    if [c[3] for c in a["cols"]] != [c[3] for c in b["cols"]]:
        print(f"首个不同轮 #{i}: anchor old={a['meta'][0]} new={b['meta'][0]}")
        print("  old draft :", [c[3] for c in a["cols"]])
        print("  new draft :", [c[3] for c in b["cols"]])
        print("  old argmax:", [c[4] for c in a["cols"]])
        print("  new argmax:", [c[4] for c in b["cols"]])
        break
else:
    print("全部轮次草稿一致 —— 说明候选空间没变（草稿头未生效？）")
PY
