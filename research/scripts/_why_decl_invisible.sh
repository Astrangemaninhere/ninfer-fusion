#!/bin/bash
# 查清为什么声明在那个 TU 里看不见：看 loader.h 的声明上下文与 decoder_state.cpp 的 include。
R=/home/user/ninfer-fusion
echo '=== loader.h:1-30（有没有 CUDA-only 守卫）==='
awk 'NR<=30 {printf "%4d| %s\n", NR, $0}' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cut -c1-125
echo
echo '=== loader.h:140-170（声明处上下文）==='
awk 'NR>=140 && NR<=170 {printf "%4d| %s\n", NR, $0}' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cut -c1-125
echo
echo '=== decoder_state.cpp 的 include 块 ==='
awk 'NR<=28 {printf "%4d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | cut -c1-125
echo
echo '=== 该头是否被别处 include、以什么方式 ==='
grep -rn 'gqa_isoquant_row_scale_loader.h' "$R/src" | head -8 | cut -c1-130
echo
echo '=== loader.h 里是否有 __CUDACC__ / ifdef 守卫 ==='
grep -nE '#if|#ifdef|#ifndef|__CUDACC__|__device__|__constant__' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | head -12 | cut -c1-120
