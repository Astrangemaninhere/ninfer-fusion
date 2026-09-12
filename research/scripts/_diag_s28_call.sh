#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 完整错误（不截断） ==="
grep -A3 'decoder_state.cpp:175' /tmp/reb_ninfer-serve_1.log 2>/dev/null | head -12
grep -B2 -A6 'is not a member of' /tmp/reb_ninfer-serve_1.log 2>/dev/null | head -20 | cut -c1-180
echo
echo "=== 那个函数在头文件里到底声明成什么 ==="
grep -n 'kv_rowscale_sidecar_apply_from_env' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cut -c1-160
echo "--- 声明处的命名空间上下文 ---"
awk '/^namespace|^} \/\/ namespace|kv_rowscale_sidecar_apply_from_env/ {print NR": "$0}' \
  "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | head -20 | cut -c1-140
echo
echo "=== 调用点现在的写法 ==="
sed -n '170,180p' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | cat -n | sed 's/^/  170+/'
