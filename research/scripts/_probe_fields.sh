#!/bin/bash
# 看现有探针（DF2SEL / DF2DBG）到底落了哪些字段，决定要补什么
R=/home/user/ninfer-fusion
echo "=== 1) DF2SEL 探针的完整打印块（dflash2_selector.cuh） ==="
grep -n -B 30 'printf("\[df2sel\]' $R/src/ops/kernel/dflash2_selector.cuh | head -55
echo
echo "=== 2) 该探针里能拿到的量：candidates / unary / scores / top_k ==="
grep -nE 'candidates\[|unary\[|scores\[|top_k|steps|pred|chosen' $R/src/ops/kernel/dflash2_selector.cuh | head -30
echo
echo "=== 3) DF2DBG（每列 verify/draft/argmax）在哪打印 ==="
grep -rn 'df2dbg' $R/src/targets/qwen3_6/impl/runtime/program_impl.h | head -8
