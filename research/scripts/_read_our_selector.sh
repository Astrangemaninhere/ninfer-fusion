#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== 我们的 dflash2_selector.cuh（全文，行数 + 内容）==='
F=$R/src/ops/kernel/dflash2_selector.cuh
wc -l "$F"
awk '{printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-150
