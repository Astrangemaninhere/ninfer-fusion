#!/bin/bash
# 细扫 pair_scale 在 0 附近（假设：pair 量级 ∝ √256 而漏了 1/√R ⇒ 正确尺度 ~1/16）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/ps_fine.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== pair_scale 细扫（0 与 1/√R 邻域）起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-12s %-9s %-22s %-10s %s\n" scale 接受率 位置剖面 提速 轮数
for s in 0 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0; do
  NINFER_DF2_PAIR_SCALE=$s timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
    > $J/dl/psf_$s.log 2>&1
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/psf_$s.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/psf_$s.log | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' $J/dl/psf_$s.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m1 'dflash2 rounds' $J/dl/psf_$s.log | grep -oE '[0-9]+')
  printf "  %-12s %-9s %-22s %-10s %s\n" "$s" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
done
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo PS_FINE_DONE
