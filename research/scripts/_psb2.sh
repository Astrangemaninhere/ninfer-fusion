#!/bin/bash
# pair_scale 大尺度扫描（不依赖 /tmp；结果落 Windows 侧持久路径）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/pair_scale_big.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== pair_scale 大尺度扫描 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-10s %-9s %-24s %s\n" scale 接受率 位置剖面 解码
for s in 1 10 100 1000 10000 1000000; do
  NINFER_DF2_PAIR_SCALE=$s timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
    > $J/dl/psb_$s.log 2>&1
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/psb_$s.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/psb_$s.log | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' $J/dl/psb_$s.log | grep -oE '[0-9.]+ tok/s')
  printf "  %-10s %-9s %-24s %s\n" "$s" "${a:-?}" "${p:-?}" "${sp:-?}"
done
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo PAIR_SCALE_BIG_DONE
