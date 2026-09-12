#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/ops/kernel/dflash2_selector.cuh
echo '=== dflash2_selector.cuh 80-256 ==='
awk 'NR>=80 && NR<=256 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
