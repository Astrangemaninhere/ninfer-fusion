#!/bin/bash
# draft-tokens sweep across the three speculative backends, through the server.
#
# Why: MTP is healthy (94/119 tok/s vs 54 baseline) while DFlash is SLOWER than
# baseline (46.1) and DFlash2 degenerates (gen=2 or gen=42). The user's point is
# that a spec path below baseline cannot be tuning — something is deeply wrong.
#   * draft_tokens=1 is the sharpest probe: one drafted token, one verify, minimal
#     overhead. If a backend is still below baseline there, its draft step itself
#     is pathologically expensive (not an acceptance/width issue).
#   * 3 and 7 show how the cost scales with width.
# Counting prompt only: it keeps generating long enough for a fair tok/s read
# (the Chinese prompt degenerates on dflash2).
set -u
PORT=${PORT:-8387}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
M=${M:-/home/user/models}
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_spec_sweep.md
MAXTOK=${MAXTOK:-192}
PROMPT='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'

one() {  # label artifact extra...
  local label=$1 art=$2
  shift 2
  local log=/home/user/sw_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$art" --port "$PORT" --max-context 8192 --no-cuda-graph --no-thinking "$@" \
    > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    echo "$label|SERVE_FAILED|$(grep -m1 -iE 'error|requires' "$log" | cut -c1-70)" >> /tmp/sweep_rows
    echo "[$label] SERVE_FAILED"; return
  fi
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' "$PROMPT" "$MAXTOK")" > /dev/null
  local line
  line=$(grep -E 'done .*speculative=' "$log" | tail -1)
  local spec gen dec
  spec=$(printf '%s' "$line" | grep -oE 'speculative=[a-z0-9_]+' | cut -d= -f2)
  gen=$(printf '%s' "$line" | grep -oE 'gen=[0-9]+' | cut -d= -f2)
  dec=$(printf '%s' "$line" | grep -oE 'decode=[0-9.]+tok/s' | cut -d= -f2)
  echo "$label|$spec|$gen|$dec" >> /tmp/sweep_rows
  echo "[$label] backend=$spec gen=$gen decode=$dec"
}

: > /tmp/sweep_rows
echo "=== draft-tokens sweep $(date +%H:%M:%S) ==="
for n in 1 3 7; do
  one "mtp_d$n"    "$M/qwen3_8_27b_nvfp4.ninfer"         --spec mtp    --draft-tokens "$n"
done
for n in 1 3 7; do
  one "dflash_d$n" "$M/qwen3_8_27b_nvfp4_dspark.ninfer"  --spec dflash --draft-tokens "$n"
done
for n in 1 3 7; do
  one "dflash2_d$n" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --spec dflash2 --draft-tokens "$n"
done
{
  echo
  echo "# draft-tokens sweep ($(date +%F' '%H:%M), counting prompt, max_tokens=$MAXTOK)"
  echo
  echo "| run | backend | gen | decode tok/s |"
  echo "|---|---|---|---|"
  while IFS='|' read -r a b c d; do echo "| $a | $b | $c | $d |"; done < /tmp/sweep_rows
  echo
  echo "baseline (no spec) on this prompt: ~54 tok/s."
  echo "draft=1 is the sharpest probe: minimal overhead, so a backend below baseline"
  echo "there has a pathological draft step rather than an acceptance/width problem."
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo SWEEP_DONE
