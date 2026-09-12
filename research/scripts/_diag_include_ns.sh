#!/bin/bash
R=/home/user/ninfer-fusion
Q=$R/_orig_quarantine
echo "=== decoder_state.cpp 的开头与第 150-180 行（找 namespace 与 include 的相对位置） ==="
sed -n '1,30p' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | cat -n | sed 's/^/  /' | cut -c1-130
echo "  ..."
sed -n '150,180p' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | cat -n | sed 's/^/  150+/' | cut -c1-130
echo
echo "=== 该文件里所有 namespace 打开/关闭与 include 的位置 ==="
grep -nE '^namespace|^\}|#include' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | head -30 | cut -c1-120
echo
echo "=== 与隔离的 .orig 对比（是谁把它改成这样的） ==="
if [ -f "$Q/src/targets/qwen3_6/impl/state/decoder_state.cpp.orig" ]; then
  diff -u "$Q/src/targets/qwen3_6/impl/state/decoder_state.cpp.orig" "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | head -40 | cut -c1-140
fi
echo
echo "=== 那个 loader 头自己的开头（看它是不是要求被放在顶层） ==="
sed -n '1,20p' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cat -n | sed 's/^/  /' | cut -c1-130
