#!/bin/bash
# 直接探查：nsys 抓 MTP k=3 与 k=1 的 nvtx range + per-kernel 统计
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:/usr/local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
for k in 3 1; do
  echo "########## nsys k=$k ##########"
  rm -f "$J/mtp_prof_k$k.nsys-rep"
  nsys profile -o "$J/mtp_prof_k$k" --force-overwrite true -t nvtx,cuda \
    --stats=true -w true \
    ./apps/ninfer "$M" --prompt "$P" --max-new 16 --max-context 4096 \
      --no-thinking --greedy --spec mtp --draft-tokens "$k" --lm-head-draft \
    > "$J/prof_k$k.log" 2>&1
  echo "  rc=$?"
  echo "  --- decode speed ---"
  grep -E 'decode speed|acceptance length' "$J/prof_k$k.log" | head -3
  echo "  --- nvtx 汇总（前 20 行）---"
  sed -n '/NVTX/,/^$/p' "$J/prof_k$k.log" | head -22
  echo "  --- CUDA GPU Kernel 汇总（前 12）---"
  sed -n '/CUDA GPU Kernel/,/^$/p' "$J/prof_k$k.log" | head -14
done
echo PROF_DONE
