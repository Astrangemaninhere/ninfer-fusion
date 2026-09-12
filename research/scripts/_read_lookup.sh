#!/bin/bash
D=/home/user/bench/pr355.diff
echo '=== pr355 触及的文件 ==='
grep -E '^\+\+\+ ' "$D" | sed 's|^+++ b/||' | sort -u | head -20
echo
echo '=== lookup 相关代码（新增行）==='
grep -nE '^\+' "$D" | grep -iE 'lookup|longest|suffix|n_?gram|history|copy|q16|q8|match' | head -30 | cut -c1-160
echo
echo '=== 关键函数定义 ==='
grep -nE '^\+.*def [a-z_]*(lookup|match|suffix|draft)[a-z_]*' "$D" | head -12 | cut -c1-150
echo
echo '=== pr366（自适应）头部 ==='
head -25 /home/user/bench/pr366.diff | cut -c1-150
