#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== launcher/dflash2_selector.h 全文 ==='
awk '{printf "%4d| %s\n", NR, $0}' "$R/src/ops/launcher/dflash2_selector.h" | cut -c1-150
echo
echo '=== dflash2_impl.h 调用处完整片段（344-349）==='
awk 'NR>=344 && NR<=349 {printf "%4d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" | cut -c1-160
