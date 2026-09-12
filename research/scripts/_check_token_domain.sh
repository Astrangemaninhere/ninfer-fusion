#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '=== 我们：TextConfig::token_domain 的定义 ==='
grep -rn 'token_domain' "$R/src/targets/qwen3_6_27b/impl/config.h" "$R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/"*.h 2>/dev/null | head -10 | cut -c1-140
echo
echo '=== 上游：token_domain 的定义 ==='
grep -rn 'token_domain' "$U/src/targets/qwen3_6_27b/impl/config.h" "$U/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/"*.h 2>/dev/null | head -10 | cut -c1-140
echo
echo '=== 上游 linear_topk 的签名与域过滤 ==='
grep -rn 'linear_topk' "$U/src/ops" "$U/include" 2>/dev/null | head -8 | cut -c1-150
f=$(grep -rl 'void linear_topk' "$U/src" 2>/dev/null | head -1)
echo "  定义/声明文件: ${f:-未找到}"
[ -n "$f" ] && grep -n -B3 -A18 'void linear_topk' "$f" | head -40 | cut -c1-150
echo
echo '=== 我们的 topk 内核是否做了域过滤（再看一遍关键行）==='
grep -n 'vocab\|token_domain\|v <' "$R/src/ops/kernel/dflash2_selector.cuh" | head -12 | cut -c1-140
