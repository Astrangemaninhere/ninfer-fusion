#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== include/ninfer/ops/dflash2_selector.h ==='
awk '{printf "%4d| %s\n", NR, $0}' "$R/include/ninfer/ops/dflash2_selector.h" | cut -c1-150
echo
echo '=== src/ops/launcher/dflash2_selector.h ==='
awk '{printf "%4d| %s\n", NR, $0}' "$R/src/ops/launcher/dflash2_selector.h" | cut -c1-150
echo
echo '=== dflash2_impl.h 里调用 selector 的那几行 ==='
awk 'NR>=325 && NR<=350 {printf "%4d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" | cut -c1-150
echo
echo '=== 我们的 TextConfig 里 output_rows / token_domain ==='
grep -rn 'output_rows\|token_domain' "$R/src/targets/qwen3_6_27b/impl/config.h" | head -6 | cut -c1-140
