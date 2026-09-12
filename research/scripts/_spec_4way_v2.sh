#!/bin/bash
# Four-way speculative comparison, v2 — with the flag the engine asked for and
# BOTH prompts, because dflash2 degenerates on one of them:
#   * `--spec mtp requires --draft-tokens in [1,5]` (the engine said so explicitly);
#   * dflash2 stops after 2 tokens on the Chinese prompt but generated a full
#     192 tokens on the counting prompt at 141.6 tok/s, so prompt matters and the
#     degenerate case must be reported as a finding, not hidden in an average.
# --no-thinking keeps the text in `content`.
set -u
PORT=${PORT:-8386}
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
M=${M:-/home/user/models}
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_spec_4way.md
MAXTOK=${MAXTOK:-192}
P1='用三句话介绍杭州的地理与历史。'
P2='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,'

req() {  # prompt tag
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' "$1" "$MAXTOK")" \
    | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); m=d["choices"][0]["message"]
    c=(m.get("content") or "").replace("\n"," ")
    print("  %s TEXT(%d): %s" % (sys.argv[1], len(c), c[:70]))
except Exception as e:
    print("  %s PARSE-FAIL: %s" % (sys.argv[1], e))' "$2"
}

run() {  # label artifact extra...
  local label=$1 art=$2
  shift 2
  local log=/home/user/s4v_$label.log
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
    echo "[$label] SERVE_FAILED: $(grep -m1 -iE 'error' "$log" | cut -c1-120)"
    echo "$label|SERVE_FAILED|-|-|-" >> /tmp/s4v_rows
    return
  fi
  echo "[$label] up"
  req "$P1" zh
  local l1 l2
  l1=$(grep -E 'done .*speculative=' "$log" | tail -1)
  req "$P2" num
  l2=$(grep -E 'done .*speculative=' "$log" | tail -1)
  printf '  zh : %s\n' "$(printf '%s' "$l1" | grep -oE 'gen=[0-9]+|decode=[0-9.]+tok/s|finish=[a-z_]+' | tr '\n' ' ')"
  printf '  num: %s\n' "$(printf '%s' "$l2" | grep -oE 'gen=[0-9]+|decode=[0-9.]+tok/s|finish=[a-z_]+' | tr '\n' ' ')"
  local spec d1 d2 g1 g2
  spec=$(printf '%s' "$l2" | grep -oE 'speculative=[a-z0-9_]+')
  g1=$(printf '%s' "$l1" | grep -oE 'gen=[0-9]+' | head -1)
  g2=$(printf '%s' "$l2" | grep -oE 'gen=[0-9]+' | head -1)
  d1=$(printf '%s' "$l1" | grep -oE 'decode=[0-9.]+tok/s' | head -1)
  d2=$(printf '%s' "$l2" | grep -oE 'decode=[0-9.]+tok/s' | head -1)
  echo "$label|$spec|zh:$g1/$d1|num:$g2/$d2|-" >> /tmp/s4v_rows
}

: > /tmp/s4v_rows
echo "=== four-way v2 $(date +%H:%M:%S) (max_tokens=$MAXTOK, --no-thinking) ==="
run plain_off  "$M/qwen3_8_27b_nvfp4.ninfer"
run plain_mtp  "$M/qwen3_8_27b_nvfp4.ninfer"         --spec mtp --draft-tokens 3
run dspark     "$M/qwen3_8_27b_nvfp4_dspark.ninfer"  --spec auto
run dflash2    "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --spec auto

{
  echo
  echo "# 四档投机对比 v2 (serve 实测 $(date +%F' '%H:%M), max_tokens=$MAXTOK, --no-thinking)"
  echo
  echo "| 配置 | backend | 中文 prompt (gen/decode) | 计数 prompt (gen/decode) |"
  echo "|---|---|---|---|"
  while IFS='|' read -r a b c d e; do echo "| $a | $b | $c | $d |"; done < /tmp/s4v_rows
  echo
  echo "读法: 中文 prompt 上 dflash2 已知有退化停止缺陷 (gen=2) ⇒ 该列不可用于速度比较;"
  echo "      计数 prompt 上各配置都能生成满 192 token ⇒ **速度只比这一列**。"
  echo "      接受率仍需 E2/S44 的计数器落地后才能给出。"
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo S4V2_DONE
