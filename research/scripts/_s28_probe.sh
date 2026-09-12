#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== ninfer_ops 源列表里 kernel/*.cu 的登记写法（找插入点） ==="
grep -nE 'ops/kernel/[a-z_]+\.cu' "$R/src/CMakeLists.txt" | head -8 | cut -c1-120
echo "  ... 列表尾部附近："
grep -n 'add_library(ninfer_ops STATIC' "$R/src/CMakeLists.txt"
awk 'NR>=66 && NR<=200 && /\)$/ {print NR": "$0; exit}' "$R/src/CMakeLists.txt"
echo
echo "=== .cu 全文（确认它可编译、无外部依赖缺失） ==="
cat -n "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.cu" | sed -n '1,20p;40,70p' | cut -c1-140
echo
echo "=== 头文件里 parse/check 的声明（声明插入点） ==="
sed -n '140,160p' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cat -n | sed 's/^/  140+/' | cut -c1-140
