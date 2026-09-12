#!/bin/bash
# 扩大 pair_scale 扫描：若边项被 unary 量级淹没，大尺度应改变选择与接受率
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/pair_scale_big.log
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || exit 3
echo "=== pair_scale 大尺度扫描 $(date '+%H:%M:%S') ==="
printf "  %-10s %-10s %-24s %s\n" scale 接受率 位置剖面 解码
for s in 1 10 100 1000 10000 1000000; do
  NINFER_DF2_PAIR_SCALE=$s timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
    > $J/dl/psb_$s.log 2>&1
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/psb_$s.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/psb_$s.log | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' $J/dl/psb_$s.log | grep -oE '[0-9.]+ tok/s')
  printf "  %-10s %-10s %-24s %s\n" "$s" "${a:-?}" "${p:-?}" "${sp:-?}"
done
echo PAIR_SCALE_BIG_DONE
