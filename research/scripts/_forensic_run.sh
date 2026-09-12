#!/bin/bash
# 逐步取证：带 [df2cand] 的投机跑 + plain 答案键，然后逐列归因
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
LOG=$J/dl/forensic.log
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || exit 3
echo "=== 逐步取证 $(date '+%F %H:%M:%S') ==="

echo "--- 1) plain（答案键） ---"
timeout 900 ./apps/ninfer "$ART" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --print-token-ids > $J/dl/fx_plain.log 2>&1
echo "  rc=$?"

echo "--- 2) spec + 候选 dump（DF2SEL=1 DF2DBG=1） ---"
NINFER_DF2SEL=1 NINFER_DF2DBG=1 timeout 1200 ./apps/ninfer "$ART" --prompt "$P" \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --print-token-ids \
  > $J/dl/fx_spec.log 2>&1
echo "  rc=$?  df2sel=$(grep -c df2sel $J/dl/fx_spec.log)  df2cand=$(grep -c df2cand $J/dl/fx_spec.log)  df2dbg=$(grep -c df2dbg $J/dl/fx_spec.log)"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length' $J/dl/fx_spec.log | sed 's/^/  /'
echo FORENSIC_RUN_DONE
