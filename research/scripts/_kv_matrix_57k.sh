#!/bin/bash
# temp: W10 — KV combination matrix at 57K on the Muse artifact (layered-KV acceptance).
# Usage: _kv_matrix_57k.sh [config ...]
#   config = <spec>                     -> POST /reload_kv on the shared server, then test
#   config = residual=<layers>@<spec>   -> fresh server with --kv-layer-storage + --kv-residual-layers
# Default: the §67 matrix subset (incl. the residual variant that needs the CLI flag).
set -u
PORT=8322
MODEL=/home/user/models/muse_glimmer_30b_nvfp4.ninfer
LOG=/home/user/kv_matrix_serve.log
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
LONGTEST=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1/tools/archkit/longtest_57k.py
CONTEXT=57344

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=("all:bf16" "0-11:e8,12-15:nvfp4" "all:nvfp4" "all:iso3" "all:int8"
           "residual=10-15@0-9:e8,10-15:nvfp4")
fi

expand_layers() {  # "10-15" -> "10,11,12,13,14,15"
  python3 /mnt/c/Users/User/Documents/ziqinzhang/_expand_layers.py "$1"
}

wait_health() {
  for _i in $(seq 1 90); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "HEALTH_OK"
      return 0
    fi
    if ! pgrep -x ninfer-serve >/dev/null; then echo "SERVE_DIED"; tail -n 6 "$LOG"; return 1; fi
  done
  echo "HEALTH_TIMEOUT"; return 1
}

start_serve() {  # $1 = extra CLI args (may be empty)
  pkill -x ninfer-serve 2>/dev/null
  sleep 3
  : > "$LOG"
  # shellcheck disable=SC2086
  setsid nohup "$BIN" "$MODEL" --port "$PORT" --kv-dtype nvfp4 --max-context 65536 \
    --no-cuda-graph $1 >> "$LOG" 2>&1 < /dev/null &
  wait_health
}

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  echo
  if [[ "$cfg" == residual=* ]]; then
    rest="${cfg#residual=}"
    layers="${rest%%@*}"
    spec="${rest#*@}"
    list=$(expand_layers "$layers")
    echo "== config [$cfg] fresh server, storage=$spec residual=$list"
    if ! start_serve "--kv-layer-storage $spec --kv-residual-layers $list"; then
      FAILED=1
      continue
    fi
    python3 "$LONGTEST" --port "$PORT" --context "$CONTEXT" || FAILED=1
    pkill -x ninfer-serve 2>/dev/null
  else
    if ! pgrep -x ninfer-serve >/dev/null; then
      echo "== starting shared server"
      if ! start_serve ""; then echo "SERVE_START_FAILED"; exit 1; fi
    fi
    echo "== config [$cfg] via reload_kv"
    python3 "$LONGTEST" --port "$PORT" --context "$CONTEXT" --reloads "$cfg" || FAILED=1
  fi
done

pkill -x ninfer-serve 2>/dev/null
echo "MATRIX_DONE failed=$FAILED"
exit $FAILED
