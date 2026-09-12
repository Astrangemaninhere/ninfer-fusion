#!/bin/bash
# 上游 dflash_impl.h vs 我们的：行数、以及语义差异摘要
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream/src/targets/qwen3_6/impl/runtime/dflash_impl.h
R=/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash_impl.h
echo "=== 行数 ==="
echo "  上游: $(wc -l < "$U")"
echo "  我们: $(wc -l < "$R")"
echo
echo '=== 上游 dflash_impl.h 里的函数/关键块 ==='
grep -nE '^(void|auto|template|DFlashFeatureSink|std::|struct|class)|propose_batch_impl|append_context_impl|DFlash2' "$U" | head -40 | cut -c1-140
echo
echo '=== 我们 dflash_impl.h 里的函数/关键块 ==='
grep -nE '^(void|auto|template|DFlashFeatureSink|std::|struct|class)|propose_batch_impl|append_context_impl|DFlash2' "$R" | head -40 | cut -c1-140
echo
echo '=== 上游里 dflash2 相关的行 ==='
grep -n 'DFlash2\|dflash2' "$U" | head -25 | cut -c1-150
