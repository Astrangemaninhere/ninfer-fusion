#!/bin/bash
# temp: W6 acceptance — decode inter-token latency under a concurrent 64K prefill, governor OFF vs ON.
# Uses the qwen3.8-27b artifact: the governor is model-agnostic, and Muse currently cannot take a
# multi-chunk prompt (see _TODO.md §89).
set -u
PORT=8326
MODEL=/home/user/models/qwen3_8_27b_nvfp4.ninfer
MODEL_ID=qwen3.8-27b
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
PROBE=/mnt/c/Users/User/Documents/ziqinzhang/_w6_bw_probe.py

wait_health() {
  local log=$1
  for _i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "HEALTH_OK"
      return 0
    fi
    if ! pgrep -x ninfer-serve >/dev/null; then echo "SERVE_DIED"; tail -n 6 "$log"; return 1; fi
  done
  echo "HEALTH_TIMEOUT"; return 1
}

run_case() {
  local label=$1 gov=$2
  local log=/home/user/w6_bw_$label.log
  pkill -x ninfer-serve 2>/dev/null
  sleep 3
  : > "$log"
  NINFER_FT_BW_GOV="$gov" setsid nohup "$BIN" "$MODEL" --port "$PORT" --kv-dtype nvfp4 \
    --max-context 69632 --max-concurrency 2 --no-cuda-graph >> "$log" 2>&1 < /dev/null &
  wait_health "$log" || return 1
  python3 "$PROBE" --port "$PORT" --context 65536 --model "$MODEL_ID" --label "$label" 2>&1 | tail -n 16
  echo "bw_lines=$(grep -c '\[ft\] bw' "$log")"
  grep '\[ft\] bw' "$log" | tail -n 4
  pkill -x ninfer-serve 2>/dev/null
  sleep 2
}

echo "########## governor OFF ##########"
run_case off 0
echo
echo "########## governor ON ##########"
run_case on 1
echo
echo "########## sanity: a normal request still answers ##########"
LOG=/home/user/w6_bw_sanity.log
: > "$LOG"
NINFER_FT_BW_GOV=1 setsid nohup "$BIN" "$MODEL" --port "$PORT" --kv-dtype nvfp4 \
  --max-context 69632 --max-concurrency 2 --no-cuda-graph >> "$LOG" 2>&1 < /dev/null &
wait_health "$LOG" || exit 1
curl -s -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 3+4? Answer with the digit only.\"}],\"max_tokens\":32,\"temperature\":0}" \
  | head -c 300
echo
pkill -x ninfer-serve 2>/dev/null
echo "W6_BW_E2E_DONE"
