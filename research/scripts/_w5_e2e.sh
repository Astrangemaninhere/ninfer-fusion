#!/bin/bash
# temp: W5 end-to-end acceptance — periodic ft-driven KV relayout on a live serve
# Expects: NINFER_FT_STATS=1 + NINFER_FT_RELOAD_SECS=30 -> after two cycles the
# stderr shows "[ft] auto-relayout -> <spec>" and generation keeps working.
set -u
PORT=8325
MODEL=/home/user/models/muse_glimmer_30b_nvfp4.ninfer
LOG=/home/user/w5_relayout.log

pkill -x ninfer-serve 2>/dev/null
sleep 2
: > "$LOG"
cd /home/user/ninfer-fusion/build
NINFER_FT_STATS=1 NINFER_FT_PERIOD=8 NINFER_FT_RELOAD_SECS=30 NINFER_FT_FULL_ATTN_LAYERS=16 \
  setsid nohup "${NINFER_SERVE_BIN:-./apps/ninfer-serve}" "$MODEL" --port "$PORT" --kv-dtype nvfp4 \
  --max-context 8192 --no-cuda-graph >> "$LOG" 2>&1 < /dev/null &

for i in $(seq 1 90); do
  sleep 2
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "HEALTH_OK $((i*2))s"
    break
  fi
  if ! pgrep -x ninfer-serve >/dev/null; then echo SERVE_DIED; tail -n 5 "$LOG"; exit 1; fi
done

G() {
  curl -s -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"muse-glimmer-30b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":24}" | head -c 120
  echo
}

echo "== warm the ft observations (3 requests) =="
for q in "2+3? digits only" "4+5? digits only" "6+7? digits only"; do G "$q"; done

echo "== waiting for two relayout cycles (75s) =="
sleep 75

echo "== relayout lines =="
grep -c "auto-relayout" "$LOG" || true
grep "auto-relayout" "$LOG" | tail -n 2 || true

echo "== generation after relayout =="
G "8+9? digits only"

echo "== serve alive? =="
pgrep -x ninfer-serve >/dev/null && echo ALIVE || echo DEAD
echo "== ft lines =="
grep -c "\[ft\] layer=" "$LOG" || true
