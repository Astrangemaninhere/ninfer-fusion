#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== 特征捕获的调用点（feature_sink 的消费）==='
grep -rn 'feature_sink\|\.layers\b.*sink\|capture' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" 2>/dev/null | head -14 | cut -c1-150
echo
echo '=== 在 text_context.h 里 sink 的定义 ==='
awk 'NR>=225 && NR<=265 {printf "%4d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | cut -c1-150
echo
echo '=== 上游：同样的 sink 消费点 ==='
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
grep -rn 'feature_sink' "$U/src/targets/qwen3_6/impl/runtime/program_impl.h" 2>/dev/null | head -8 | cut -c1-150
