#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/ops/launcher/dflash2_selector.cu
echo "=== 行数 ==="; wc -l "$F"
echo '=== 全文 ==='
awk '{printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
