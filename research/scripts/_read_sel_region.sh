#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== dflash2_selector.cuh 240-300 ==="
sed -n '240,300p' $R/src/ops/kernel/dflash2_selector.cuh
echo
echo "=== 该文件的行数与包含它的 TU ==="
wc -l $R/src/ops/kernel/dflash2_selector.cuh
grep -rln 'dflash2_selector.cuh' $R/src 2>/dev/null | sed "s|$R/||" | head -10
