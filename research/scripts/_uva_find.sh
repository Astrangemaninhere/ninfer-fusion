#!/bin/bash
V=/home/user/vllm029/lib/python3.12/site-packages/vllm
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) UVA 检查的 raise 点（单包内定点 grep 长字符串） ==="
grep -rn 'UVA is not available' $V 2>/dev/null | head -5
echo "--- 该文件上下文 ---"
F=$(grep -rl 'UVA is not available' $V 2>/dev/null | head -1)
[ -n "$F" ] && grep -n -B 12 -A 4 'UVA is not available' "$F" | head -40
echo
echo "=== 2) 是否与 UVA/umap 相关的开关（env） ==="
F2=$(grep -rl 'uva\|UVA' $V/config/*.py 2>/dev/null | head -2)
for f in $F2; do echo "--- $f"; grep -nE 'uva|UVA' "$f" | head -8; done
echo
echo "=== 3) 日志里 UVA 前后的行 ==="
grep -n -B 6 'UVA is not available' $J/dl/vref_df2.err | head -20
