#!/bin/bash
# EXACTNESS-ONLY (no export, no 25 GB copy): does speculative decode reproduce plain
# decode's token sequence? Greedy speculative decode is exact iff the verifier accepts
# drafts by the target's own argmax; any divergence in the emitted text is a
# correctness bug in the verify path (shared by DFlash2 and DSpark).
# Also prints the first divergence offset so the failing position is visible.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
M=/home/user/models
PORT=8391
BIN=/home/user/ninfer-fusion/build/apps/ninfer-serve
DF2=$M/qwen3_8_27b_nvfp4_dflash2.ninfer
PLAIN=$M/qwen3_8_27b_nvfp4.ninfer
LOG=$J/dl/df2_exactness.log
MAXTOK=96
P_ZH='用三句话介绍杭州的地理与历史。'
P_NUM='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'

exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== exactness run $(date '+%F %H:%M:%S')  bin=$(stat -c %y "$BIN" | cut -c1-16) ==="
free -m | head -2

# memory watchdog: never let this run push the box into the reboot zone
( while true; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt 700 ]; then
      echo "  [watchdog] MemAvailable=${avail}MB < 700MB -> killing serve"
      for p in $(pgrep -f ninfer-serve); do grep -qa ninfer-serve /proc/$p/cmdline && kill -TERM "$p"; done
      break
    fi
    sleep 10
  done ) &
WD=$!

kill_serve() {
  for p in $(pgrep -f ninfer-serve 2>/dev/null); do
    grep -qa ninfer-serve /proc/$p/cmdline 2>/dev/null || continue
    kill -TERM "$p" 2>/dev/null
  done
  sleep 4
}

serve() {  # artifact extra...
  local art=$1; shift
  local tag=${1:-nospec}
  local log=/home/user/ex_$(basename "$art" .ninfer)_$tag.log
  kill_serve
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph "$@" \
    > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 150); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    pgrep -f ninfer-serve >/dev/null || break
  done
  if [ $ok -ne 1 ]; then echo "[serve $art $*] FAILED"; tail -4 "$log"; return 1; fi
  echo "$log"
}

ask() {  # tag prompt log
  local tag=$1 prompt=$2 log=$3
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' "$prompt" "$MAXTOK")" \
    > /tmp/ex_resp_$tag.json
  python3 - "$tag" <<'PY'
import json, sys
tag = sys.argv[1]
try:
    d = json.load(open("/tmp/ex_resp_%s.json" % tag))
    m = d["choices"][0]["message"]["content"]
except Exception as e:
    print("  [%s] PARSE-FAIL %s" % (tag, e)); m = ""
open("/tmp/ex_text_%s.txt" % tag, "w").write(m)
print("  [%s] %d chars: %r" % (tag, len(m), m[:70].replace("\n", " ")))
PY
  grep -E 'done .*gen=' "$log" | tail -1 | sed 's/^.*ninfer-serve: //' | cut -c1-200 | sed "s/^/  [$tag engine] /"
}

cmp_report() {  # name a.txt b.txt
  local name=$1 a=$2 b=$3
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then echo "[$name] missing text"; return; fi
  if cmp -s "$a" "$b"; then
    echo "[$name] *** IDENTICAL *** ($(wc -c < "$a") bytes) -> verifier is exact"
  else
    python3 - "$name" "$a" "$b" <<'PY'
import sys
name, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
A = open(pa, encoding="utf-8", errors="replace").read()
B = open(pb, encoding="utf-8", errors="replace").read()
n = min(len(A), len(B))
i = next((k for k in range(n) if A[k] != B[k]), n)
print("[%s] *** DIFFERENT *** first divergence at char %d of %d/%d" % (name, i, len(A), len(B)))
print("    common prefix : %r" % A[:i][-60:])
print("    plain  next 60: %r" % A[i:i+60])
print("    spec   next 60: %r" % B[i:i+60])
PY
  fi
}

l1=$(serve "$PLAIN") && ask plain_zh "$P_ZH" "$l1" && ask plain_num "$P_NUM" "$l1"
l2=$(serve "$DF2" --spec auto) && ask spec_zh "$P_ZH" "$l2" && ask spec_num "$P_NUM" "$l2"
l3=$(serve "$DF2" --spec auto) && ask spec2_zh "$P_ZH" "$l3" && ask spec2_num "$P_NUM" "$l3"
kill_serve

echo
echo "=== verdict ==="
cmp_report "plain-vs-spec zh"  /tmp/ex_text_plain_zh.txt  /tmp/ex_text_spec_zh.txt
cmp_report "plain-vs-spec num" /tmp/ex_text_plain_num.txt /tmp/ex_text_spec_num.txt
cmp_report "spec-vs-spec  zh"  /tmp/ex_text_spec_zh.txt   /tmp/ex_text_spec2_zh.txt
cmp_report "spec-vs-spec  num" /tmp/ex_text_spec_num.txt  /tmp/ex_text_spec2_num.txt
kill "$WD" 2>/dev/null
echo "DF2_EXACTNESS_DONE"
