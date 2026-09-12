#!/bin/bash
# 串行：① BF16 头 artifact（最后一道门已开）② 带轮次标记的 16x16 dump
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
OLD=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
NEW=/home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/serial2.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
pkill -9 -x ninfer 2>/dev/null; sleep 2
cd "$R/build" || exit 3
echo "=== 串行第二轮 起 $(date '+%m-%d %H:%M:%S') ==="

echo "--- ① BF16 头 artifact 短生成冒烟 ---"
timeout 600 ./apps/ninfer "$NEW" --prompt "$P" --max-new 16 --max-context 1024 \
  --no-thinking --greedy > $J/dl/s2_bf16_plain.log 2>&1
echo "  rc=$?"
grep -m1 'error' $J/dl/s2_bf16_plain.log | head -c 200; echo
grep -m1 'decode speed' $J/dl/s2_bf16_plain.log | sed 's/^/  /'

echo "--- ② BF16 头 artifact dflash2 接受率 ---"
timeout 900 ./apps/ninfer "$NEW" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --spec dflash2 > $J/dl/s2_bf16_d2.log 2>&1
grep -m1 -E 'dflash2 acceptance rate' $J/dl/s2_bf16_d2.log | sed 's/^/  /'
grep -m1 -E 'dflash2 accepted by pos' $J/dl/s2_bf16_d2.log | sed 's/^/  /'
grep -m1 'decode speed' $J/dl/s2_bf16_d2.log | sed 's/^/  /'

echo "--- ③ 带轮次标记的 dump ---"
rm -f $J/dl/df2scores_*.bin 2>/dev/null
NINFER_DF2SCORES=1 timeout 900 ./apps/ninfer "$OLD" --prompt "$P" \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph \
  > $J/dl/s2_scores.log 2>&1
echo "  rc=$?"
echo "  scores 文件=$(ls $J/dl/df2scores_scores_*.bin 2>/dev/null | wc -l)  front 文件=$(ls $J/dl/df2scores_front_*.bin 2>/dev/null | wc -l)  anch 文件=$(ls $J/dl/df2scores_anch_*.bin 2>/dev/null | wc -l)"
ls -l $J/dl/df2scores_front_1.bin $J/dl/df2scores_anch_1.bin 2>/dev/null | awk '{print "  " $5, $9}'
grep -m1 'dflash2 accepted by pos' $J/dl/s2_scores.log | sed 's/^/  /'
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo SERIAL2_DONE
