#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/include/ninfer/ops/swa.h
echo '=== 行数 ==='; wc -l "$F" 2>/dev/null || echo '  该路径不存在'
echo '=== swa.h 全文（契约）==='
[ -f "$F" ] && awk '{printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
echo
echo '=== 找 swa 的其它声明位置 ==='
grep -rn 'void swa(' "$R/src" "$R/include" 2>/dev/null | head -5 | cut -c1-140
