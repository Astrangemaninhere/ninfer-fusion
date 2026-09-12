#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '=== 我们：特征捕获的调用点（DFlashFeatureSink / prefill_features / consume_prefill）==='
grep -rn 'DFlashFeatureSink\|prefill_features\|consume_prefill\|feature_sink' "$R/src" 2>/dev/null | grep -v '\.orig' | head -20 | cut -c1-150
echo
echo '=== 上游：同名的东西 ==='
grep -rn 'DFlashFeatureSink\|prefill_features\|consume_prefill\|feature_sink' "$U/src" 2>/dev/null | head -20 | cut -c1-150
echo
echo '=== 我们：consume_prefill 的定义/实现 ==='
f=$(grep -rl 'consume_prefill' "$R/src" 2>/dev/null | grep -v '\.orig' | head -1)
echo "  文件: ${f:-未找到}"
