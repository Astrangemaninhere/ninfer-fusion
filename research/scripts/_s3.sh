#!/bin/bash
# 重跑 dump（同时开 DF2SCORES 与 DF2DBG，二者同轮同号）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
cd /home/user/ninfer-fusion/build || exit 3
rm -f "$J"/dl/df2scores_*.bin
NINFER_DF2SCORES=1 NINFER_DF2DBG=1 timeout 900 ./apps/ninfer \
  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph \
  > "$J/dl/s3_scores.log" 2>&1
echo "rc=$?"
echo "dbg 轮数=$(grep -c 'df2dbg] row=' "$J/dl/s3_scores.log")"
echo "dump 组数=$(ls "$J"/dl/df2scores_scores_*.bin 2>/dev/null | wc -l)"
grep -m1 'accepted by pos' "$J/dl/s3_scores.log" || true
