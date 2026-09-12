#!/bin/bash
# 从 nsys 报告里取内核耗时排行（k=3 vs k=1 对比）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
for k in 3 1; do
  echo "########## k=$k 内核耗时 Top ##########"
  awk '/cuda_gpu_kern_sum/{f=1} f&&/^ Time/{g=1} g' "$J/prof_k$k.log" 2>/dev/null | head -26
  echo
  echo "--- k=$k 其它区间汇总（nvtx / osrt）---"
  sed -n '/nvtx_sum\|NVTX/,/^\[/p' "$J/prof_k$k.log" 2>/dev/null | head -14
  echo
done
echo TOP_DONE
