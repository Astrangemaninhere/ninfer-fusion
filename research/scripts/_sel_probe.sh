#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== DF2SEL 探针是否存在及其格式 ==="
grep -rn 'NINFER_DF2SEL' $R/src --include=*.cuh --include=*.h --include=*.cu 2>/dev/null | grep -v '\.orig' | head
echo "--- 格式串 ---"
grep -rn 'df2sel' $R/src/ops/kernel/dflash2_selector.cuh 2>/dev/null | head -20
echo
echo "=== 是否还有 DF2_PAIR_SCALE / DF2DBG ==="
grep -rln 'NINFER_DF2' $R/src 2>/dev/null | head
