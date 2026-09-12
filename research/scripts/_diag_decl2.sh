#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 头文件里 check/parse 的声明（找插入点） ==="
grep -nE 'kv_rowscale_sidecar_(parse|check)|^\} // namespace' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cut -c1-120
echo
echo "=== 该 .cu 是否在 CMake 源列表里（否则链接会缺符号） ==="
grep -rn 'gqa_isoquant_row_scale_loader' "$R/src/CMakeLists.txt" | head -4 | cut -c1-130
echo
echo "=== ninfer_engine 是否链接 ninfer_ops ==="
grep -nE 'target_link_libraries\(ninfer_engine|ninfer_ops' "$R/src/CMakeLists.txt" | head -6 | cut -c1-130
echo
echo "=== .cu 里 apply 函数的结尾（返回值语义） ==="
sed -n '38,60p' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.cu" | cat -n | sed 's/^/  38+/' | cut -c1-140
