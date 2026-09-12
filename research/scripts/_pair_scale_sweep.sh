#!/bin/bash
# 确认 pair_scale 门控存在，然后扫几个值看接受率
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/pair_scale_sweep.log
exec > >(tee -a "$LOG") 2>&1
echo "=== pair_scale 门控检查 $(date '+%H:%M:%S') ==="
grep -rn 'NINFER_DF2_PAIR_SCALE\|pair_scale' $R/src/ops/kernel/dflash2_selector.cuh $R/src/ops/launcher/dflash2_selector.cu 2>/dev/null | head -8
grep -rn 'pair_scale' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | head -5

cd "$R/build" || exit 3
echo
echo "=== 扫 pair_scale ==="
printf "  %-8s %-12s %s\n" scale 接受率 位置剖面
for s in "" 0.5 1.0 2.0 4.0; do
  tag=${s:-default}
  if [ -z "$s" ]; then
    timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --prompt "$P" \
      --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
      > $J/dl/ps_$tag.log 2>&1
  else
    NINFER_DF2_PAIR_SCALE=$s timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
      --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
      > $J/dl/ps_$tag.log 2>&1
  fi
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/ps_$tag.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/ps_$tag.log | sed 's/.*pos *//')
  printf "  %-8s %-12s %s\n" "$tag" "${a:-?}" "${p:-?}"
done
echo PAIR_SCALE_SWEEP_DONE
