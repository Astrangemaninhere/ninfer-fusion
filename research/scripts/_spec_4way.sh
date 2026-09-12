#!/bin/bash
# Four-way speculative-decoding comparison THROUGH THE SERVER (user asked for it):
#   plain_off   plain artifact, no spec        (baseline)
#   plain_mtp   plain artifact, --spec mtp     (the artifact's auto backend)
#   dspark      dspark artifact, --spec auto   (SpeculativeBackend::DFlash)
#   dflash2     dflash2 artifact, --spec auto  (SpeculativeBackend::DFlash2)
#
# --no-thinking keeps the answer in `content` (last run's empty content was just
# thinking-mode routing), so the text comparison is apples-to-apples.
# The engine prints no acceptance counter yet (a patch for that is in flight), so
# this measures what exists: decode tok/s, wall, and output sanity.
set -u
PORT=${PORT:-8385}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
M=${M:-/home/user/models}
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_spec_4way.md
MAXTOK=${MAXTOK:-192}

req() {
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' "$1" "$MAXTOK")" \
    | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); m=d["choices"][0]["message"]
    c=(m.get("content") or "").replace("\n"," ")
    print("  TEXT(%d): %s" % (len(c), c[:90]))
except Exception as e:
    print("  PARSE-FAIL:", e)'
}

run() {  # label artifact extra...
  local label=$1 art=$2
  shift 2
  local log=/home/user/s4w_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$art" --port "$PORT" --max-context 8192 --no-cuda-graph --no-thinking "$@" \
    > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 150); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    echo "[$label] SERVE_FAILED"; tail -3 "$log"; echo "$label|SERVE_FAILED||" >> /tmp/s4w_rows; return
  fi
  echo "[$label] up"
  req '用三句话介绍杭州的地理与历史。'
  # second prompt: the counting one that made dflash2 stop at gen=2 (verify defects
  # show up here first because the answer is long and highly predictable).
  req '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'
  local line
  line=$(grep -E 'done .*speculative=' "$log" | tail -1)
  echo "  $line" | cut -c1-400
  local dec gen spec
  dec=$(printf '%s' "$line" | grep -oE 'decode=[0-9.]+tok/s' | head -1)
  gen=$(printf '%s' "$line" | grep -oE 'gen=[0-9]+' | head -1)
  spec=$(printf '%s' "$line" | grep -oE 'speculative=[a-z0-9_]+' | head -1)
  echo "$label|$dec|$gen|$spec" >> /tmp/s4w_rows
}

: > /tmp/s4w_rows
echo "=== four-way spec comparison $(date +%H:%M:%S) ==="
run plain_off    "$M/qwen3_8_27b_nvfp4.ninfer"
run plain_mtp    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp
run dspark       "$M/qwen3_8_27b_nvfp4_dspark.ninfer"   --spec auto
run dflash2      "$M/qwen3_8_27b_nvfp4_dflash2.ninfer"  --spec auto
pkill -f "$BIN" 2>/dev/null || true
sleep 3

# --- per-position acceptance: the position-resolved profile is what distinguishes
# "draft is weak everywhere" (p_0 already low) from "position/convention mismatch"
# (healthy p_0, collapsing tail). apps/ninfer prints `<backend> accepted by pos`.
CLI=/home/user/ninfer-fusion/build/apps/ninfer
Q='用三句话介绍杭州的地理与历史。'
cli_pos() {  # label artifact spec-args...
  local label=$1 art=$2
  shift 2
  local log=/home/user/s4wpos_$label.log
  echo "=== CLI [$label] $(date +%H:%M:%S) ==="
  ( cd /home/user/ninfer-fusion/build && timeout 900 "$CLI" "$art" --prompt "$Q" \
      --max-new 96 --max-context 4096 --no-thinking "$@" > "$log" 2>&1 )
  echo "  rc=$?  log=$log"
  grep -hiE 'acceptance|accepted by pos|tok/round|decode' "$log" | head -8 | cut -c1-200
  { echo; echo "## CLI per-position [$label] $(date +%H:%M)"; grep -hiE 'acceptance|accepted by pos|tok/round' "$log" | head -8; } >> "$OUT"
}
cli_pos cli_mtp3    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp    --draft-tokens 3
cli_pos cli_dspark  "$M/qwen3_8_27b_nvfp4_dspark.ninfer"   --spec dflash --draft-tokens 7
cli_pos cli_dflash2 "$M/qwen3_8_27b_nvfp4_dflash2.ninfer"  --spec dflash2


# --- long-context needle check: 32 needles (the 8-needle run was too noisy: the
# baseline itself scored 5/8 at 16K but 7/8 at 32K). MTP must MATCH the baseline;
# a speculative path that is fast and forgetful is a quality regression.
LT=${J_LONGTEST:-/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/longtest_57k.py}
lc() {  # label artifact spec-args...
  local label=$1 art=$2
  shift 2
  local log=/home/user/lc_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  ( cd /home/user/ninfer-fusion/build && setsid nohup "$BIN" "$art" --port "$PORT" \
      --max-context 32768 --no-cuda-graph --no-thinking "$@" > "$log" 2>&1 < /dev/null & )
  local ok=0
  for _i in $(seq 1 180); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    pgrep -f "$BIN" >/dev/null || break
  done
  if [ $ok -eq 0 ]; then echo "[$label] SERVE_FAILED"; echo "$label|SERVE_FAILED|-|-" >> /tmp/s4w_lc; return; fi
  for ctx in 16384 32768; do
    local hits
    hits=$(timeout 1500 python3 "$LT" --port "$PORT" --context "$ctx" --needles 32 \
             --model qwen3.8-27b 2>&1 | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
    echo "[$label ctx=$ctx] ${hits:-<no result>}"
    echo "$label|$ctx|${hits#needle hits }|$(grep -oE 'speculative=[a-z0-9_]+' "$log" | tail -1)" >> /tmp/s4w_lc
  done
}
J_LONGTEST=${J_LONGTEST:-/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/longtest_57k.py}
: > /tmp/s4w_lc
if [ -f "$J_LONGTEST" ]; then
  echo "=== long-context 32-needle check $(date +%H:%M:%S) ==="
  lc lc_base    "$M/qwen3_8_27b_nvfp4.ninfer"
  lc lc_mtp3    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp --draft-tokens 3
  lc lc_dflash2 "$M/qwen3_8_27b_nvfp4_dflash2.ninfer"  --spec auto
  pkill -f "$BIN" 2>/dev/null || true
  {
    echo
    echo "## 长上下文 needle 32 发 ($(date +%F' '%H:%M))"
    echo
    echo "| 配置 | ctx | hits | backend |"
    echo "|---|---|---|---|"
    while IFS='|' read -r a b c d; do echo "| $a | $b | $c | $d |"; done < /tmp/s4w_lc
    echo
    echo "判读: 投机档命中数必须与基线相同; 低于基线即质量回归 (掉针=检索失败)。"
  } >> "$OUT"
else
  echo "  (longtest tool absent: $J_LONGTEST)"
fi

{
  echo "# 四档投机对比 (serve 实测, $(date +%F' '%H:%M), max_tokens=$MAXTOK, --no-thinking)"
  echo
  echo "| 配置 | decode | gen | speculative |"
  echo "|---|---|---|---|"
  while IFS='|' read -r a b c d; do echo "| $a | $b | $c | $d |"; done < /tmp/s4w_rows
  echo
  echo "注: 引擎尚无接受率计数 (E2/S44 补丁在做), 故本表只有速度与合法性;"
  echo "    MTU/DFlash/DFlash2 的差异需要计数到位后才能归因 (草稿宽度/接受率/草稿成本)。"
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo S4W_DONE
