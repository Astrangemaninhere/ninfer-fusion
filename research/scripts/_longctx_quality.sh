#!/bin/bash
# Long-context QUALITY comparison: baseline vs MTP (and MTP with a wider draft).
#
# The speed numbers say MTP is healthy (+72%/+120%); the user is right to doubt
# that without a quality check, because a speculative path can be fast and wrong.
# Needle-in-a-haystack at 16K and 32K is the cheap, quantitative probe: needles
# are placed at known depths and the serve is asked to retrieve them, so a
# corrupted verify/draft shows up as missed needles, not as a vibe.
#
# Queued behind the draft-tokens sweep (waits for its SWEEP_DONE marker) so only
# one GPU job runs at a time.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
PORT=${PORT:-8388}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
M=${M:-/home/user/models}
LONGTEST=$J/ninfer-fusion-repo/tools/archkit/longtest_57k.py
OUT=$J/_collab/M_longctx_quality.md
LOG=$J/dl/longctx_quality.log

exec >> "$LOG" 2>&1
echo "=== long-context quality run $(date +%F' '%H:%M:%S) ==="
t0=$(date +%s)
while ! grep -q 'SWEEP_DONE' "$J/dl/sweep.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo "timeout waiting for sweep; proceeding"; break; }
done
echo "sweep finished; starting"

run() {  # label artifact spec-args...
  local label=$1 art=$2
  shift 2
  local log=/home/user/lq_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$art" --port "$PORT" --max-context 32768 --no-cuda-graph --no-thinking "$@" \
    > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 180); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    echo "$label|SERVE_FAILED|-|-" >> /tmp/lq_rows
    echo "[$label] SERVE_FAILED"; return
  fi
  for ctx in 16384 32768; do
    local hits
    hits=$(timeout 900 python3 "$LONGTEST" --port "$PORT" --context "$ctx" --needles 8 \
             --model qwen3.8-27b 2>&1 | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
    echo "[$label ctx=$ctx] ${hits:-<no result>}"
    echo "$label|$ctx|${hits#needle hits }|$(grep -oE 'speculative=[a-z0-9_]+' "$log" | tail -1)" >> /tmp/lq_rows
  done
}

: > /tmp/lq_rows
run baseline "$M/qwen3_8_27b_nvfp4.ninfer"
run mtp3     "$M/qwen3_8_27b_nvfp4.ninfer" --spec mtp --draft-tokens 3
run mtp5     "$M/qwen3_8_27b_nvfp4.ninfer" --spec mtp --draft-tokens 5

{
  echo
  echo "# 长上下文质量对照 (needle-in-a-haystack, $(date +%F' '%H:%M))"
  echo
  echo "| 配置 | ctx | needle hits | backend |"
  echo "|---|---|---|---|"
  while IFS='|' read -r a b c d; do echo "| $a | $b | $c | $d |"; done < /tmp/lq_rows
  echo
  echo "判读: MTP 的命中数必须与基线**相同**; 任一 ctx 下低于基线即质量回归"
  echo "      (掉针 = 检索失败 = 输出不可信, 与速度无关)。"
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo LONGCTX_DONE
