#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo "=== 文件行数 ==="
wc -l "$F"
echo
echo '=== 100-205 行（特征投影 / 上下文 K/V）==='
awk 'NR>=100 && NR<=205 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-150
