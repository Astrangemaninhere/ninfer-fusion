#!/bin/bash
# temp: the shipped default KV table vs the safe alternatives, at 32K needles.
set -u
PORT=8368
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
MODEL=${KV_MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LONGTEST=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1/tools/archkit/longtest_57k.py

run() {  # label extra-args...
  local label=$1
  shift
  pkill -x ninfer-serve 2>/dev/null
  sleep 3
  local log=/home/user/kvdef_$label.log
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$MODEL" --port "$PORT" --max-context 32768 --no-cuda-graph \
    "$@" > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -x ninfer-serve >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then echo "[$label] SERVE_FAILED"; tail -2 "$log"; return; fi
  local hits
  hits=$(python3 "$LONGTEST" --port "$PORT" --context 32768 --needles 8 --model qwen3.8-27b 2>&1 \
    | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
  echo "[$label] $hits"
}

# Shipped default (no kv flags at all): the model's registered table.
run shipped_default
# The safe alternative: e8 only on the verified low layers.
run safe_low "kv-dtype nvfp4 --kv-layer-storage 0-7:e8"
# Plain nvfp4 everywhere.
run plain_nvfp4 --kv-dtype nvfp4
pkill -x ninfer-serve 2>/dev/null
echo KV_DEFAULT_PROBE_DONE
