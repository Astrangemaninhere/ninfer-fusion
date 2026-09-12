#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 1) mtp_round.cuh 的 ar_valid_columns 与 ar_hidden ==="
grep -nE 'ar_valid_columns|ar_hidden|valid_columns' "$R/src/ops/kernel/mtp_round.cuh" 2>/dev/null | head -20 | cut -c1-150
echo
echo "--- 现场（A1 说的 :47 附近） ---"
sed -n '36,60p' "$R/src/ops/kernel/mtp_round.cuh" 2>/dev/null | cat -n | sed 's/^/  36+/' | cut -c1-150
echo
echo "=== 2) 该 kernel 的调用方（谁传 valid_columns / 语义是什么） ==="
grep -rn 'ar_valid_columns\|mtp_round' "$R/src/ops/wrapper/" "$R/src/targets/qwen3_6/impl/runtime/mtp_impl.h" 2>/dev/null | head -12 | cut -c1-150
echo
echo "=== 3) 编译进度 ==="
tail -4 /mnt/c/Users/User/Documents/ziqinzhang/dl/rebuild_after_s35.log | cut -c1-140
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
