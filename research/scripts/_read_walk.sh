#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/ops/kernel/sampling_device.cuh
echo "=== 文件行数 ==="; wc -l "$F"
echo '=== dflash2_selector_walk_kernel 定义与上下文 ==='
N=$(grep -n 'dflash2_selector_walk_kernel' "$F" | head -1 | cut -d: -f1)
echo "  定义行: $N"
awk -v s=$((N-5)) -v e=$((N+75)) 'NR>=s && NR<=e {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
