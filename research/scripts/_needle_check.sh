#!/bin/bash
# temp: Muse needle retrieval at 57K on the current binary (W10 quality gate + output sanity).
set -u
PORT=8333
CTX=${1:-57344}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
LOG=/home/user/needle_check.log
MAXCTX=$((CTX + 8192))
pkill -x ninfer-serve 2>/dev/null
sleep 2
cd /home/user/ninfer-fusion/build || exit 1
setsid nohup "$BIN" /home/user/models/muse_glimmer_30b_nvfp4.ninfer --port "$PORT" \
  --kv-dtype nvfp4 --max-context "$MAXCTX" --no-cuda-graph > "$LOG" 2>&1 < /dev/null &
for _i in $(seq 1 90); do
  sleep 2
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then echo HEALTH_OK; break; fi
  if ! pgrep -x ninfer-serve >/dev/null; then echo SERVE_DIED; tail -5 "$LOG"; exit 1; fi
done
python3 /mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1/tools/archkit/longtest_57k.py --port "$PORT" \
  --context "$CTX" --needles 4 2>&1 | tail -8
echo "--- serve log tail"
grep -E 'error|bad_alloc|exceed|error std' "$LOG" | tail -4
pkill -x ninfer-serve 2>/dev/null
echo NEEDLE_CHECK_DONE
