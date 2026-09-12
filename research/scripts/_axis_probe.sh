#!/bin/bash
# 判别漂移的累积轴：按「轮」还是按「token」。
# 同一 prompt，比较 K=1 与 K=7 两种轮次结构：从探针日志还原每轮的 licensed token 序列，
# 与 plain 流比对，找出「第几轮 / 第几个生成 token」首次偏离。
#   两者首次偏离的**轮数**相同 ⇒ 按轮累积（轮次状态/回滚 → S_B 方向）
#   两者首次偏离的**token 数**相同 ⇒ 按 token 累积（GDN 递推本身 → S_A 方向）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/axis_probe.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== drift axis probe $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --print-token-ids > $J/dl/ax_plain.log 2>&1
grep -E '^tokens[[:space:]]+generated ids' $J/dl/ax_plain.log | tail -1 | \
    sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/ax_plain.ids
for k in 1 7; do
  NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --spec dflash2 --draft-tokens $k \
      > $J/dl/ax_k$k.log 2> $J/dl/ax_k$k.probe
  echo "  k=$k probe=$(wc -l < $J/dl/ax_k$k.probe)"
done

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib, re
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
round_re = re.compile(r"row=\d+ anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                      r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
col_re = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")

def parse(p):
    out, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        m = round_re.search(line)
        if m:
            cur = {"accepted": int(m.group(6)), "count": int(m.group(7)), "cols": []}
            out.append(cur); continue
        m = col_re.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return out

def plain_ids(p):
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            return [int(x) for x in line.split("generated ids", 1)[1].split()]
    return []

plain = plain_ids(D / "ax_plain.log")
print(f"plain n={len(plain)}")
for k in (1, 7):
    rounds = parse(D / f"ax_k{k}.probe")
    if not rounds:
        print(f"  K={k}: NO PROBE"); continue
    # 滞后校正：本轮真实 accepted = 下一行的 accepted
    for i, r in enumerate(rounds):
        r["own"] = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
    lic, idx, first = [], 0, None
    for n, r in enumerate(rounds):
        if r["own"] is None or not r["cols"]:
            continue
        a = max(0, r["own"])
        toks = [c["draft"] for c in r["cols"][:a]]                  # 被接受的草稿
        toks.append(r["cols"][min(a, len(r["cols"]) - 1)]["argmax"])  # correction/bonus
        for t in toks:
            if idx >= len(plain):
                break
            if t != plain[idx] and first is None:
                first = (n + 1, idx)          # 第 n+1 轮、第 idx 个生成 token
            idx += 1
    print(f"  K={k}: rounds={len(rounds)} 生成={idx} "
          f"首次偏离=第 {first[0] if first else '-'} 轮 / 第 {first[1] if first else '-'} 个 token")
PY
echo AXIS_PROBE_DONE
