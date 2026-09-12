#!/bin/bash
# Muse acceptance THROUGH THE SERVER (user's requirement: 测试得直接拉 server 测).
#
# Judgement semantics fixed after A/S42 showed the first version could FALSE-PASS:
# it counted only `NAN` lines, so an explicit geometry refusal (no nvfp4 for
# head_dim=128) and a cudaErrorIllegalAddress crash both scored "zero NAN" ⇒ PASS.
# A PASS now requires ALL of:
#   * the serve started and /health answered,
#   * the HTTP request returned a parseable body with non-empty text,
#   * the text is not mojibake/garbage,
#   * zero NAN lines in the serve log,
#   * no abort / illegal-address / CUDA error line in the serve log.
# A serve that refuses to start is reported as REFUSED (its own state — for
# Muse+nvfp4 after the 256 fix that is the *expected honest* outcome, not a pass).
set -u
PORT=${PORT:-8372}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
MODEL=${KV_MODEL:-/home/user/models/muse_glimmer_30b_nvfp4.ninfer}
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_muse_serve_accept.md
PROMPT='1, 2, 3, 4, 5, 6,'
declare -A STATE NAN TEXT

[ -x "$BIN" ] || { echo "FATAL: serve binary missing: $BIN"; echo MUSE_SERVE=FAIL; exit 2; }
[ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; echo MUSE_SERVE=FAIL; exit 2; }

one() {  # dtype
  local kv=$1
  local log=/home/user/muse_srv_$kv.log
  pgrep -af ninfer-serve | sed 's/^/  kill? /'     # 铁律②: 杀前先看 cmdline
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  NINFER_HEADDBG=1 setsid nohup "$BIN" "$MODEL" --port "$PORT" --max-context 4096 \
    --no-cuda-graph --kv-dtype "$kv" > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    if grep -qiE 'requires head_dim|not ported|invalid argument' "$log"; then
      STATE[$kv]=REFUSED
    else
      STATE[$kv]=SERVE_FAILED
    fi
    NAN[$kv]=$(grep -c 'NAN' "$log" || true)
    TEXT[$kv]="<no request: ${STATE[$kv]}>"
    echo "[$kv] ${STATE[$kv]}  nan_lines=${NAN[$kv]}"
    grep -m1 -iE 'requires head_dim|not ported|error' "$log" | cut -c1-140 | sed 's/^/      /'
    return
  fi
  local body resp
  body=$(printf '{"model":"muse-glimmer-30b","messages":[{"role":"user","content":"%s"}],"max_tokens":8,"temperature":0,"stream":false}' "$PROMPT")
  resp=$(curl -s -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
         -H 'Content-Type: application/json' -d "$body")
  TEXT[$kv]=$(printf '%s' "$resp" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].replace("\n"," ")[:80])
except Exception as e:
    print("<parse-fail: %s>" % e)' 2>/dev/null)
  NAN[$kv]=$(grep -c 'NAN' "$log" || true)
  local cuda_err
  cuda_err=$(grep -ciE 'illegal address|Aborted|cudaError|terminate called' "$log" || true)
  # classification: crash > mojibake > nan > clean
  local t="${TEXT[$kv]}"
  if [ "$cuda_err" -gt 0 ]; then STATE[$kv]=CRASH
  elif printf '%s' "$t" | grep -q '<parse-fail'; then STATE[$kv]=BAD_RESPONSE
  elif printf '%s' "$t" | grep -qP '[\x{FFFD}]|\xEF\xBF\xBD' 2>/dev/null; then STATE[$kv]=GARBAGE
  elif [ "${NAN[$kv]}" -gt 0 ]; then STATE[$kv]=NAN
  elif [ -z "$(printf '%s' "$t" | tr -d '[:space:]')" ]; then STATE[$kv]=EMPTY
  else STATE[$kv]=CLEAN
  fi
  echo "[$kv] ${STATE[$kv]}  nan_lines=${NAN[$kv]}  cuda_err_lines=$cuda_err  text=\"$t\""
}

one bf16
one nvfp4

echo "=== verdict ==="
verdict=PASS
for kv in bf16 nvfp4; do
  s="${STATE[$kv]:-MISSING}"
  echo "  $kv: state=$s nan_lines=${NAN[$kv]:-} text=\"${TEXT[$kv]:-}\""
  case "$s" in
    CLEAN) ;;
    REFUSED) [ "$verdict" = PASS ] && verdict=REFUSED ;;   # honest refusal, not a pass
    *) verdict=FAIL ;;
  esac
done
echo MUSE_SERVE=$verdict
{
  echo "# Muse serve-path acceptance ($(date +%F' '%H:%M))"
  echo
  echo "| dtype | state | nan_lines | text |"
  echo "|---|---|---|---|"
  for kv in bf16 nvfp4; do echo "| $kv | ${STATE[$kv]:-} | ${NAN[$kv]:-} | ${TEXT[$kv]:-} |"; done
  echo
  echo "PASS requires state=CLEAN (started, HTTP body parsed, non-garbage text, 0 NaN, no CUDA error)."
  echo "REFUSED = the engine explicitly refused this KV dtype for this geometry (expected for"
  echo "nvfp4 on head_dim=128 until that kernel is ported) — it is NOT a pass."
  echo "verdict: **$verdict**"
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo MUSE_SERVE_DONE
