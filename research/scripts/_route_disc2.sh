#!/bin/bash
# 补跑 plain（带 --print-token-ids），然后对 w2/w8 两份已有探针日志算列0一致率
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 3
timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
  --print-token-ids > "$J/dl/rd_plain_ids.log" 2>&1
echo "plain_ids rc=$? ids=$(grep -c 'generated ids' $J/dl/rd_plain_ids.log)"

python3 - <<'PY'
import pathlib, re
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

def parse(p):
    rounds, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m: cur = {"accepted": int(m.group(7)), "count": int(m.group(8)), "cols": []}; rounds.append(cur); continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "verify": int(m.group(3)), "argmax": int(m.group(5))})
    return rounds

plain = plain_ids(D / "rd_plain_ids.log")
print(f"plain ids={len(plain)}")
for tag in ("w2", "w8"):
    rd = parse(D / f"rd_{tag}.log")
    if not rd: print(f"[{tag}] 无输出"); continue
    cur = 0
    for r in rd:
        r["s"] = None
        if r["cols"]:
            head = r["cols"][0]["verify"]
            for i in range(cur, len(plain)):
                if plain[i] == head: r["s"] = i; cur = i; break
    for i, r in enumerate(rd):
        r["own"] = rd[i+1]["accepted"] if i+1 < len(rd) else None
    W = max((len(r["cols"]) for r in rd), default=0)
    n = [0]*W; ok = [0]*W
    for r in rd:
        if r["s"] is None or r["own"] is None: continue
        for c in r["cols"]:
            j = c["col"]; t = r["s"] + j + 1
            if j > max(0, r["own"]) or t >= len(plain) or c["verify"] == -1: continue
            n[j] += 1; ok[j] += (c["argmax"] == plain[t])
    acc = sum(r["accepted"] for r in rd if r["accepted"] >= 0)
    cnt = sum(r["count"] for r in rd if r["count"] > 0)
    print(f"[{tag}] 可锚定={sum(1 for r in rd if r['s'] is not None)}/{len(rd)}  列0 {ok[0]}/{n[0]} = "
          f"{100.0*ok[0]/n[0] if n[0] else 0:.1f}%  总 {sum(ok)}/{sum(n)} = "
          f"{100.0*sum(ok)/sum(n) if sum(n) else 0:.1f}%  接受率={100.0*acc/cnt if cnt else 0:.1f}%")
PY
