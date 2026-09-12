#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== 全树搜 dflash2_selector_walk_kernel ==='
grep -rn 'dflash2_selector_walk_kernel' "$R/src" 2>/dev/null | head -10 | cut -c1-140
echo
f=$(grep -rl '__global__ void dflash2_selector_walk_kernel' "$R/src" 2>/dev/null | head -1)
echo "  定义文件: ${f:-未找到}"
if [ -n "$f" ]; then
  n=$(grep -n '__global__ void dflash2_selector_walk_kernel' "$f" | head -1 | cut -d: -f1)
  echo "  定义行: $n"
  awk -v s="$n" -v e="$((n+65))" 'NR>=s && NR<=e {printf "%4d| %s\n", NR, $0}' "$f" | cut -c1-150
fi
