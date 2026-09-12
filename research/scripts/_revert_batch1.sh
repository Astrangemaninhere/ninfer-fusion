#!/bin/bash
R=/home/user/ninfer-fusion
Q=$R/_orig_quarantine
echo "=== [1] 回退 S24 窗口接线（layouts_impl.h，guard 在非模板上下文里无效 + designator 顺序错） ==="
cp "$Q/src/targets/qwen3_6/impl/runtime/layouts_impl.h.orig" "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h"
echo "  已回退 layouts_impl.h（原文件保留在隔离区）"
grep -c 'is_swa_attention' "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" | sed 's/^/  live 里 is_swa_attention 引用数: /'
echo
echo "=== [2] 回退 S28（decoder_state.cpp 的半落地调用，连 include 一起） ==="
cp "$Q/src/targets/qwen3_6/impl/state/decoder_state.cpp.orig" "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp"
grep -c 'rowscale' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | sed 's/^/  live 里 rowscale 引用数: /'
echo
echo "=== [3] text_context_impl.h 的结构性错误现场（116-126 行 + 往前找不闭合） ==="
sed -n '100,126p' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | cat -n | sed 's/^/  100+/' | cut -c1-140
echo
echo "=== [4] 该文件是否也有 .orig（有没有备份） ==="
ls -l "$Q/src/targets/qwen3_6/impl/runtime/" 2>/dev/null | awk '{print "  ", $5, $NF}'
ls -l "$Q/src/targets/qwen3_6/impl/runtime/text_context_impl.h.orig" 2>/dev/null || echo "  （无 text_context_impl.h.orig）"
echo
echo "=== [5] 该文件里 include 与 namespace 的相对位置（同类地雷自查） ==="
grep -nE '^namespace|^\}|^#include' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | head -20 | cut -c1-120
