#!/bin/bash
# e8 pre/post-rotation scale fix end-to-end check: 32K needle acceptance for the
# configs that degraded before the fix (see _TODO.md 96/116). CPU-side analysis
# lives in _e8_fixsim.py. This script now judges: a run that never produced a
# count is a FAIL, and the pre-fix degradation pattern (front window 0-7 losing
# to plain) is called out explicitly instead of being left to the reader.
set -u
PORT=8369
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
MODEL=${KV_MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LONGTEST=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/longtest_57k.py
declare -A HITS=()

[ -x "$BIN" ] || { echo "FATAL: serve binary missing: $BIN"; echo E8_VERDICT=FAIL; exit 2; }
[ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; echo E8_VERDICT=FAIL; exit 2; }

run() {  # label extra-args...
  local label=$1
  shift
  pgrep -af ninfer-serve | sed 's/^/  kill? /'    # 铁律②: 杀之前先看 cmdline
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  local log=/home/user/e8fix_$label.log
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$MODEL" --port "$PORT" --max-context 32768 --no-cuda-graph \
    "$@" > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then echo "[$label] SERVE_FAILED"; tail -2 "$log"; HITS[$label]="SERVE_FAILED"; return; fi
  local line
  line=$(python3 "$LONGTEST" --port "$PORT" --context 32768 --needles 8 --model qwen3.8-27b 2>&1 \
    | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
  HITS[$label]="${line#needle hits }"
  echo "[$label] $line"
}

# NOTE (2026-09-10): these three lines used to pass ONE quoted string
# ("kv-dtype nvfp4 --kv-layer-storage 0-7:e8"), which the server rejected as
# `unknown argument: kv-dtype nvfp4 --kv-layer-storage 0-7:e8` — the e8 tiers
# were never actually exercised. Pass separate args (and the leading --).
run plain_nvfp4 --kv-dtype nvfp4
run e8_0_7 --kv-dtype nvfp4 --kv-layer-storage 0-7:e8
run e8_8_15 --kv-dtype nvfp4 --kv-layer-storage 8-15:e8
run e8_14_15 --kv-dtype nvfp4 --kv-layer-storage 14-15:e8

echo "=== verdict ==="
verdict=PASS
for k in plain_nvfp4 e8_0_7 e8_8_15 e8_14_15; do
  echo "  $k = ${HITS[$k]:-MISSING}"
  [[ "${HITS[$k]:-}" =~ ^[0-9]+/[0-9]+$ ]] || { verdict=FAIL; echo "  !! $k 未产出计数"; }
done
if [ "$verdict" = PASS ]; then
  p=${HITS[plain_nvfp4]%%/*}
  for k in e8_0_7 e8_8_15 e8_14_15; do
    echo "  $k - plain = $(( ${HITS[$k]%%/*} - p ))"
  done
  # 前窗 0-7 是 e8 的验证窗口: 修复后它不该输给 plain (允许 1 针噪声)。
  if [ $(( ${HITS[e8_0_7]%%/*} - p )) -lt -1 ]; then
    verdict=REVIEW
    echo "  !! e8 0-7 掉针低于 plain-1 (${HITS[e8_0_7]} vs ${HITS[plain_nvfp4]})"
  fi
fi
echo E8_VERDICT=$verdict
pgrep -af ninfer-serve | sed 's/^/  kill? /'
pkill -f "$BIN" 2>/dev/null || true
echo E8_POSTFIX_DONE
