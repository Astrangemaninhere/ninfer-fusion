#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== 1) 编译/验证状态 ==="
tail -4 "$J/dl/rebuild_after_s35.log" | cut -c1-140
ls -l --time-style=+%H:%M $R/build/apps/ninfer $R/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo "--- 验证链 ---"
tail -10 "$J/dl/offset_family_verify.log" 2>/dev/null | cut -c1-175
echo
echo "=== 2) ar_valid_columns 的消费者链（决定正确参照量） ==="
sed -n '190,215p' "$R/src/targets/qwen3_6/impl/runtime/mtp_impl.h" | cat -n | sed 's/^/  190+/' | cut -c1-155
echo
echo "=== 3) mtp_forward_core / ar 步的 valid 用途（内核侧） ==="
grep -rn 'valid_columns' "$R/src/ops/kernel/mtp*.cuh" "$R/src/ops/wrapper/mtp_round.cpp" 2>/dev/null | head -12 | cut -c1-150
