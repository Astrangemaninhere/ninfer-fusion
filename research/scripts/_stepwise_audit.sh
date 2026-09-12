#!/bin/bash
# 逐步体检：把 verify 每列的 (pos, argmax) 与 plain 流按绝对位置对齐，找出第一个漂移列。
# 判据：
#   verify.argmax[pos] == plain[pos] 全场成立  => target logits 没错，问题在 accept/记账
#   某 pos 起恒不等                        => verify 的 logits 本身与 plain 不同（状态/参数）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/stepwise_audit.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== stepwise audit $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

echo "--- (1) plain（无投机） ---"
timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --print-token-ids > $J/dl/sw_plain.log 2>&1
grep -E '^tokens[[:space:]]+generated ids' $J/dl/sw_plain.log | tail -1 | \
    sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/sw_plain.ids
echo "    plain n=$(wc -w < /tmp/sw_plain.ids)  prompt_tokens=$(grep -oE 'prompt tokens[[:space:]]+[0-9]+' $J/dl/sw_plain.log | tail -1 | grep -oE '[0-9]+')"

echo "--- (2) dflash2 + 列探针 ---"
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec dflash2 > $J/dl/sw_stdout.log 2> $J/dl/sw_probe.log
echo "    probe lines=$(wc -l < $J/dl/sw_probe.log)"

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib, re
round_re = re.compile(r"\[df2dbg\] row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) "
                      r"in_extent=(-?\d+) out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
col_re = re.compile(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
rounds, cur = [], None
for line in pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/sw_probe.log") \
        .read_text(errors="replace").splitlines():
    m = round_re.search(line)
    if m:
        cur = {"anchor": int(m.group(2)), "frontier": int(m.group(3)), "valid": int(m.group(4)),
               "in_extent": int(m.group(5)), "accepted": int(m.group(7)), "cols": []}
        rounds.append(cur); continue
    m = col_re.search(line)
    if m and cur is not None:
        cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                            "verify": int(m.group(3)), "draft": int(m.group(4)),
                            "argmax": int(m.group(5))})
plain = []
for line in pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/sw_plain.log") \
        .read_text(errors="replace").splitlines():
    if line.startswith("tokens") and "generated ids" in line:
        plain = [int(x) for x in line.split("generated ids", 1)[1].split()]

# prompt length: plain positions start at prompt_len
prompt_len = None
for line in pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/sw_plain.log") \
        .read_text(errors="replace").splitlines():
    if "prompt tokens" in line:
        prompt_len = int(line.split()[-1])
print(f"rounds={len(rounds)}  plain={len(plain)}  prompt_len={prompt_len}")
print()
print("按绝对位置对齐：verify.argmax[pos]  vs  plain[pos - prompt_len]")
seen = {}
order = []
for r in rounds:
    for c in r["cols"]:
        p = c["pos"]
        if p not in seen:
            seen[p] = (c["argmax"], c["verify"], r["in_extent"], r["accepted"], c["col"])
            order.append(p)
order.sort()
first_bad = None
bad = 0
for p in order:
    argmax, verify_tok, ext, acc, col = seen[p]
    idx = p - prompt_len
    if 0 <= idx < len(plain):
        if argmax != plain[idx]:
            bad += 1
            if first_bad is None:
                first_bad = (p, idx, verify_tok, argmax, plain[idx], ext, acc, col)
print(f"覆盖位置数={len(order)}  与 plain 不一致的位置数={bad}")
if first_bad:
    p, idx, vt, am, pl, ext, acc, col = first_bad
    print(f"首个漂移列: pos={p} (生成序 {idx}) col={col} 投喂 token={vt} "
          f"verify.argmax={am}  plain={pl}  in_extent={ext} accepted={acc}")
    lo = max(0, idx - 4)
    print(f"  plain[{lo}:{idx+4}] = {plain[lo:idx+4]}")
    print(f"  该轮 col0..3 = "
          f"{[c['argmax'] for c in rounds[0]['cols'][:4]] if rounds else []}")
else:
    print("全部覆盖位置一致 ⇒ target logits 无偏差，问题在 accept/记账侧")
PY
echo STEPWISE_DONE
