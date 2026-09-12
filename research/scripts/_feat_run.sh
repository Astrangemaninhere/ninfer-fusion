#!/bin/bash
# 跑一次特征链 dump（短生成足够，dump 发生在第一次 append）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 3
exec > >(tee -a "$J/dl/feat_run.log") 2>&1
echo "=== 特征链 dump $(date '+%H:%M:%S') ==="
rm -f $J/dl/feat_features.bin $J/dl/feat_projected.bin $J/dl/feat_context.bin
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/feat_stdout.log 2>&1
echo "  rc=$?"
grep -E 'df2feat|dflash2 acceptance' $J/dl/feat_stdout.log | head -4
ls -l --time-style=+%H:%M $J/dl/feat_*.bin 2>/dev/null | awk '{print "  ", $5, $6, $7}'
echo FEAT_RUN_DONE
