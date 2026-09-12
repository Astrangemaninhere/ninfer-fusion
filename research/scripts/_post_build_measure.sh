#!/bin/bash
# POST-BUILD MEASUREMENT (fires only when the rebuild that contains Patch A has landed).
# Linear and explicit on purpose: one serve at a time, no shared state, no $@ juggling.
#
#  plain  : reference greedy text (no --spec)      -> defines the exact sequence
#  dflash2: must reproduce plain byte-for-byte + acceptance
#  mtp3   : the known-good reference backend        -> is non-exactness shared?
#
# Waits for: no make/nvcc/ptxas running AND ninfer-serve newer than the patch time.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
M=/home/user/models
BIN=/home/user/ninfer-fusion/build/apps/ninfer-serve
DF2=$M/qwen3_8_27b_nvfp4_dflash2.ninfer
PLAIN=$M/qwen3_8_27b_nvfp4.ninfer
LOG=$J/dl/postbuild_measure.log
OUT=$J/_collab/M_patchA_effect.md
PORT=8392
MAXTOK=96
REF_EPOCH=${REF_EPOCH:-0}     # binary must be newer than this; set by the launcher
P_ZH='用三句话介绍杭州的地理与历史。'
P_NUM='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'

exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== post-build measure armed $(date '+%F %H:%M:%S'); need bin newer than $(date -d @$REF_EPOCH '+%F %H:%M:%S' 2>/dev/null || echo "$REF_EPOCH") ==="

t0=$(date +%s)
while :; do
  busy=0
  pgrep -x make >/dev/null 2>&1 && busy=1
  pgrep -x nvcc >/dev/null 2>&1 && busy=1
  pgrep -x ptxas >/dev/null 2>&1 && busy=1
  pgrep -f 'bin/nvcc|cc1plus|cicc' >/dev/null 2>&1 && busy=1
  mtime=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
  if [ "$busy" -eq 0 ] && [ "$mtime" -gt "$REF_EPOCH" ]; then break; fi
  now=$(date +%s)
  if [ $(( now - t0 )) -gt 21600 ]; then echo "TIMEOUT after 6h (busy=$busy mtime=$mtime)"; echo POSTBUILD_TIMEOUT; exit 1; fi
  if [ $(( (now - t0) % 600 )) -lt 60 ]; then
    echo "  [$(date +%H:%M:%S)] waiting: busy=$busy bin_mtime=$(date -d @$mtime '+%H:%M:%S' 2>/dev/null)"
  fi
  sleep 60
done
echo "=== build clear at $(date '+%H:%M:%S'); bin=$(stat -c %y "$BIN" | cut -c1-19) size=$(stat -c %s "$BIN") ==="

kill_serve() {
  for p in $(pgrep -f ninfer-serve 2>/dev/null); do
    grep -qa ninfer-serve /proc/$p/cmdline 2>/dev/null || continue
    kill -TERM "$p" 2>/dev/null
  done
  sleep 4
}

one() {   # name artifact max_tokens extra-args...   (tags: <name>_zh, <name>_num)
  local name=$1 art=$2 mt=$3; shift 3
  local log=/home/user/pb_$name.log
  kill_serve
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  # --no-thinking is mandatory here: without it the answer lands in reasoning_content and
  # `content` comes back empty, so the exactness diff would compare "" vs "" and report
  # IDENTICAL - proving nothing. (Measured 17:44 on the new binary: content "", reasoning set.)
  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph --no-thinking "$@" \
    > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 150); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    pgrep -f ninfer-serve >/dev/null || break
  done
  if [ $ok -ne 1 ]; then echo "[$name] SERVE_FAILED"; tail -4 "$log"; echo "$name|SERVE_FAILED|-|-|-" >> /tmp/pb_rows; return 1; fi
  # two requests
  for pair in zh num; do
    if [ "$pair" = zh ]; then pr="$P_ZH"; else pr="$P_NUM"; fi
    curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' "$pr" "$mt")" \
      > /tmp/pb_resp_${name}_$pair.json
    python3 -c "
import json,sys
try:
    d=json.load(open('/tmp/pb_resp_${name}_$pair.json'))
    m=d['choices'][0]['message']['content']
except Exception as e:
    m=''
    print('  [${name}_$pair] PARSE-FAIL', e)
open('/tmp/pb_text_${name}_$pair.txt','w').write(m)
print('  [${name}_$pair] %d chars' % len(m))
"
    grep -E 'done .*gen=' "$log" | tail -1 | cut -c1-230 | sed "s/^/  [${name}_$pair engine] /"
  done
  local line; line=$(grep -E 'done .*speculative=' "$log" | tail -1)
  echo "$name|$(printf '%s' "$line" | grep -oE '[0-9.]+tok/round' | head -1)|$(printf '%s' "$line" | grep -oE '\(([0-9.]+)%\)' | head -1)|$(printf '%s' "$line" | grep -oE 'gen=[0-9]+' | tail -1)|$(printf '%s' "$line" | grep -oE 'finish=[a-z_]+' | tail -1)" >> /tmp/pb_rows
}

: > /tmp/pb_rows
echo
echo "### stage 1/3 plain (reference)"
one plain  "$PLAIN" "$MAXTOK"
echo "### stage 2/3 dflash2 (--spec auto)"
one dflash2 "$DF2"  "$MAXTOK" --spec auto
echo "### stage 3/3 mtp3 (--spec mtp --draft-tokens 3)"
one mtp3   "$DF2"  "$MAXTOK" --spec mtp --draft-tokens 3
kill_serve

echo
echo "=== EXACTNESS (spec vs plain, same prompt, greedy) ==="
python3 - <<'PY' >> /home/user/pb_verdict.txt
import pathlib
def rd(p):
    f = pathlib.Path(p)
    return f.read_text(encoding="utf-8", errors="replace") if f.exists() else None
for pair in ("zh", "num"):
    base = rd("/tmp/pb_text_plain_%s.txt" % pair)
    for name in ("dflash2", "mtp3"):
        other = rd("/tmp/pb_text_%s_%s.txt" % (name, pair))
        if base is None or other is None:
            print("[%s %s] missing text" % (name, pair)); continue
        if not base or not other:
            # An empty side is NOT agreement: no content came back (thinking mode, parse
            # failure, or a rejected request), so it cannot support any conclusion.
            print("[%s %s] INVALID - empty text on at least one side (%d vs %d chars); NOT evidence"
                  % (name, pair, len(base), len(other))); continue
        if base == other:
            print("[%s %s] IDENTICAL (%d chars) -> verifier exact" % (name, pair, len(base)))
        else:
            n = min(len(base), len(other)); i = next((k for k in range(n) if base[k] != other[k]), n)
            print("[%s %s] DIFFERENT at char %d (%d vs %d chars)" % (name, pair, i, len(base), len(other)))
            print("    common : %r" % base[:i][-40:])
            print("    plain  : %r" % base[i:i+40])
            print("    spec   : %r" % other[i:i+40])
PY
cat /home/user/pb_verdict.txt

{
  echo
  echo "## Patch A 效果实测 ($(date '+%F %H:%M'))"
  echo
  echo "binary: \`$(stat -c %y "$BIN" | cut -c1-19)\` size=$(stat -c %s "$BIN")"
  echo
  echo "| run | tok/round | accept% | gen | finish |"
  echo "|---|---|---|---|---|"
  while IFS='|' read -r a b c d e; do [ -n "$a" ] && echo "| $a | $b | $c | $d | $e |"; done < /tmp/pb_rows
  echo
  echo '```'
  cat /home/user/pb_verdict.txt
  echo '```'
  echo
  echo "判读: IDENTICAL ⇒ 投机路径与 plain 逐字节一致（verify 精确）; DIFFERENT ⇒ 仍有 verify 正确性 bug。"
  echo "接受率: 修前 dflash2 counting=0.0% / zh=50.9%; MTP 参考带 43-56%。"
} >> "$OUT"
echo POSTBUILD_DONE
