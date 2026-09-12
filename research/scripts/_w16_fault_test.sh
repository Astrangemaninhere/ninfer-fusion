#!/bin/bash
# temp: W16 acceptance — lane-level request failure and invariant engine stop.
set -u
PORT=8327
MODEL=/home/user/models/muse_glimmer_30b_nvfp4.ninfer
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}

start_serve() {  # $1 = inject mode, $2 = log path
  pkill -x ninfer-serve 2>/dev/null
  sleep 3
  : > "$2"
  NINFER_FAULT_INJECT="$1" NINFER_FAULT_INJECT_MIN_ID=2 setsid nohup "$BIN" "$MODEL" \
    --port "$PORT" --kv-dtype nvfp4 --max-context 8192 --no-cuda-graph >> "$2" 2>&1 < /dev/null &
  for _i in $(seq 1 90); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "HEALTH_OK"
      return 0
    fi
    if ! pgrep -x ninfer-serve >/dev/null; then echo "SERVE_DIED"; tail -n 6 "$2"; return 1; fi
  done
  echo "HEALTH_TIMEOUT"; return 1
}

G() {
  curl -s -m 120 -w '\nHTTP %{http_code}\n' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"muse-glimmer-30b","messages":[{"role":"user","content":"3+4? digits only"}],"max_tokens":8}' \
    | head -c 400
}

echo "########## A. request-domain fault: one request fails, engine keeps serving ##########"
start_serve request_once /home/user/w16_request.log || exit 1
python3 /mnt/c/Users/User/Documents/ziqinzhang/_w16_fault_probe.py --port "$PORT"
RC_A=$?
pgrep -x ninfer-serve >/dev/null && echo "ALIVE" || echo "DEAD"
echo "== engine log lines"; grep '\[engine\] request id=' /home/user/w16_request.log | tail -2

echo
echo "########## B. invariant fault: engine stops, /health reports failed + reason ##########"
start_serve invariant_once /home/user/w16_invariant.log || exit 1
echo "== req1 (expect failure)"; G
echo "== /health (expect 503 failed)"; curl -s -m 5 -w ' HTTP %{http_code}\n' "http://127.0.0.1:$PORT/health"
grep -E 'injected invariant' /home/user/w16_invariant.log | tail -2
echo "== POST /recover (W16 P1)"
curl -s -m 300 -w ' HTTP %{http_code}\n' -X POST "http://127.0.0.1:$PORT/recover"
echo "== /health after recover (expect 200 ok)"
curl -s -m 5 -w ' HTTP %{http_code}\n' "http://127.0.0.1:$PORT/health"
echo "== req2 after recover (expect a normal answer)"; G
pkill -x ninfer-serve 2>/dev/null
echo "A_RC=$RC_A"
echo "W16_FAULT_TEST_DONE"
