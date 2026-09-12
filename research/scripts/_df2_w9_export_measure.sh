#!/bin/bash
# Build a dflash2 artifact from TODAY'S trained draft and measure its acceptance.
#
# Why this is the decisive experiment for "is 50.9% the draft or the mechanism":
# the artifact we have been measuring was exported on Aug 26; today's W9 training
# (step_001200 and counting) has never been inside an artifact. Same engine, same
# prompts, only the draft weights change — so the acceptance delta attributes the
# shortfall.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
M=/home/user/models
PORT=8389
BIN=/home/user/ninfer-fusion/build/apps/ninfer-serve
BASE=$M/qwen3_8_27b_nvfp4.ninfer
CKPT=$J/data/dflash2_ckpts/step_001200.pt
NEW=$M/qwen3_8_27b_nvfp4_dflash2_w9s1200.ninfer
OUT=$J/_collab/M_df2_w9_draft.md
LOG=$J/dl/df2_w9_export.log

exec >> "$LOG" 2>&1
echo "=== dflash2 export from trained draft $(date +%F' '%H:%M:%S) ==="
df -h /home | tail -1

# wait for the long-context quality run to release the GPU
t0=$(date +%s)
while ! grep -q 'LONGCTX_DONE' "$J/dl/longctx_quality.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo "timeout waiting for longctx; proceeding"; break; }
done
echo "gpu free; building artifact"

echo "--- patch_dflash2 (dry-run first) ---"
# The tool imports `tools.artifact.container`, so it must run from the repo root
# with that root on PYTHONPATH (plain `python3 <path>` fails with
# ModuleNotFoundError: No module named 'tools').
cd "$J/ninfer-fusion-repo" || { echo "repo missing"; exit 3; }
export PYTHONPATH="$J/ninfer-fusion-repo"
python3 tools/convert/qwen3_8_27b/patch_dflash2.py \
  --src "$BASE" --ckpt "$CKPT" --out "$NEW" --dry-run 2>&1 | tail -6
echo "--- patch_dflash2 (real) ---"
python3 tools/convert/qwen3_8_27b/patch_dflash2.py \
  --src "$BASE" --ckpt "$CKPT" --out "$NEW" 2>&1 | tail -8
ls -l "$NEW" 2>/dev/null | cut -c1-110

echo "--- verify_patch ---"
python3 tools/convert/qwen3_8_27b/verify_patch.py \
  --src "$BASE" --out "$NEW" 2>&1 | tail -10 || python3 tools/convert/qwen3_8_27b/verify_patch.py --help 2>&1 | head -6

measure() {  # label artifact prompt
  local label=$1 art=$2 prompt=$3
  local log=/home/user/w9_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$art" --port $PORT --max-context 8192 --no-cuda-graph --no-thinking \
    --spec auto > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 150); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    pgrep -f "$BIN" >/dev/null || break
  done
  [ $ok -eq 1 ] || { echo "[$label] SERVE_FAILED"; tail -3 "$log"; echo "$label|SERVE_FAILED|-|-|-" >> /tmp/w9_rows; return; }
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":192,"temperature":0,"stream":false}' "$prompt")" > /dev/null
  local line; line=$(grep -E 'done .*speculative=' "$log" | tail -1)
  echo "[$label] $(printf '%s' "$line" | grep -oE 'gen=[0-9]+|speculative=[a-z0-9_]+ [0-9.]+tok/round \([0-9.]+%\)|finish=[a-z_]+' | tr '\n' ' ')"
  echo "$label|$(printf '%s' "$line" | grep -oE '[0-9.]+tok/round \([0-9.]+%\)' | tr '\n' ' ')|$(printf '%s' "$line" | grep -oE 'gen=[0-9]+' | cut -d= -f2)|$(printf '%s' "$line" | grep -oE 'finish=[a-z_]+' | cut -d= -f2)|$(printf '%s' "$line" | grep -oE 'decode=[0-9.]+tok/s' | cut -d= -f2)" >> /tmp/w9_rows
}

: > /tmp/w9_rows
echo "--- measure old draft vs new draft (same engine, same prompts) ---"
measure new_num "$NEW"  '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'
measure old_num "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,'
measure new_zh  "$NEW"  '用三句话介绍杭州的地理与历史。'
measure old_zh  "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" '用三句话介绍杭州的地理与历史。'

{
  echo
  echo "# 用今天训练出来的草稿重造 artifact 后的接受率 ($(date +%F' '%H:%M))"
  echo
  echo "| run | accept (tok/round (%) ) | gen | finish | decode |"
  echo "|---|---|---|---|---|"
  while IFS='|' read -r a b c d e; do echo "| $a | $b | $c | $d | $e |"; done < /tmp/w9_rows
  echo
  echo "ckpt = $CKPT (W9, step 1200); base = $(basename $BASE); new = $(basename $NEW)"
  echo "读法: old vs new 的差值就是**草稿权重**的贡献; 若两者相近, 则瓶颈在机制而非训练。"
  echo "目标 (用户口径): 位置1 接受率 ~0.9; 当前 dflash2 每轮 4.55/7 = 50.9%。"
} >> "$OUT"
pkill -f "$BIN" 2>/dev/null || true
echo DF2_W9_EXPORT_DONE
