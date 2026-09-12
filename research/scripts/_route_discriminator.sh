#!/bin/bash
# 关键判别实验（零改码）：verify 的块宽 W 决定它走哪条 split-K 路由。
#   W=8（--draft-tokens 7）=> T=8  => Prompt 路由（与 plain T=1 的 SmallT 不同）
#   W=2（--draft-tokens 1）=> T=2  => SmallT 路由（与 plain T=1 同一家族）
# 若 W=2 时列0 一致率远高于 W=8，则"路由不同⇒归约顺序/精度不同"是机制本身；
# 若两者相同，则机制在别处（掩码/量化/dequant）。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
D=$J/dl
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
cd "$R/build" || exit 3
echo "=== binary $(stat -c '%y %s' $BIN | cut -c1-19,21-32) ==="

run(){ # $1=tag  $2..=extra args
  local tag=$1; shift
  timeout 900 "$BIN" "$M" --prompt "$P" --max-new 96 --max-context 4096 \
     --no-thinking --greedy "$@" > "$D/rd_$tag.log" 2>&1
  echo "  $tag rc=$? size=$(stat -c %s $D/rd_$tag.log)"
}

echo "--- 1) plain（对照） ---"
run plain
echo "--- 2) W=2 (draft-tokens 1) + 探针 ---"
NINFER_DF2DBG=1 run w2 --spec dflash2 --draft-tokens 1
grep -m2 -E 'error|require|invalid' "$D/rd_w2.log" | head -2
echo "--- 3) W=8 (draft-tokens 7) + 探针 ---"
NINFER_DF2DBG=1 run w8 --spec dflash2 --draft-tokens 7
grep -m2 -E 'error|require|invalid' "$D/rd_w8.log" | head -2

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

plain = plain_ids(D / "rd_plain.log")
print(f"plain ids={len(plain)}")
for tag, W in (("w2", 2), ("w8", 8)):
    rd = parse(D / f"rd_{tag}.log")
    if not rd: print(f"[{tag}] 无探针输出（可能 --draft-tokens 被拒）"); continue
    cur = 0
    for r in rd:
        r["s"] = None
        if r["cols"]:
            head = r["cols"][0]["verify"]
            for i in range(cur, len(plain)):
                if plain[i] == head: r["s"] = i; cur = i; break
    for i, r in enumerate(rd):
        r["own"] = rd[i+1]["accepted"] if i+1 < len(rd) else None
    n = o = 0
    for r in rd:
        if r["s"] is None or r["own"] is None: continue
        for c in r["cols"]:
            j = c["col"]; t = r["s"] + j + 1
            if j > max(0, r["own"]) or t >= len(plain) or c["verify"] == -1: continue
            if j == 0:
                n += 1; o += (c["argmax"] == plain[t])
    acc = sum(r["accepted"] for r in rd if r["accepted"] >= 0)
    cnt = sum(r["count"] for r in rd if r["count"] > 0)
    print(f"[{tag}] 轮数={len(rd)} 列0 一致率 = {o}/{n} = {100.0*o/n if n else 0:.1f}%   "
          f"接受率 = {acc}/{cnt} = {100.0*acc/cnt if cnt else 0:.1f}%")
print("判读：W=2 显著高于 W=8 ⇒ 路由差异即机制；两者相近 ⇒ 机制在掩码/量化/dequant。")
PY
echo ROUTE_DISCRIMINATOR_DONE
