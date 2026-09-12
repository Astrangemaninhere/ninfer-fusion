#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== feature_projection 的用法 ==='
grep -rn 'feature_projection' $R/src --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | grep -v '\.orig' | head -10 | cut -c1-150
echo
echo '=== context_norm 的用法 ==='
grep -rn 'context_norm' $R/src --include=*.h --include=*.cpp 2>/dev/null | grep -v '\.orig' | head -8 | cut -c1-140
echo
echo '=== 目标层选择：target_layer_ids / context_layers / hs layers ==='
grep -rnE 'target_layer|context_layer|hs_layer|dflash_context|context_rows|feature_rows' $R/src --include=*.h --include=*.cpp --include=*.cuh 2>/dev/null | grep -v '\.orig' | head -20 | cut -c1-155
echo
echo '=== dflash_context_impl.h 的结构（前 60 行）==='
F=$(ls $R/src/targets/qwen3_6/impl/runtime/dflash_context_impl.h 2>/dev/null)
[ -n "$F" ] && awk 'NR<=60 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-150
