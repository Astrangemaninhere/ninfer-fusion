#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '=== 我们 dflash2_selector.cpp（wrapper，全文）==='
awk '{printf "%4d| %s\n", NR, $0}' "$R/src/ops/wrapper/dflash2_selector.cpp" | cut -c1-150
echo
echo '=== 我们的 token_domain 取值 ==='
grep -rn 'kTokenDomain\|token_domain' "$R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/"*.h 2>/dev/null | head -8 | cut -c1-140
grep -rn 'kTokenDomain' "$R/include" 2>/dev/null | head -5 | cut -c1-140
echo
echo '=== 上游 rmsnorm_pack_tail 的声明（语义）==='
grep -rn -A 20 'void rmsnorm_pack_tail' "$U/src" "$U/include" 2>/dev/null | head -32 | cut -c1-150
