#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6_27b/impl/config.h
echo '=== 27b config.h 85..145（含 dflash / dflash2 两段）==='
awk 'NR>=85 && NR<=145 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-150
echo
echo '=== 全树 target_feature_layers 的所有出现 ==='
grep -rn 'target_feature_layers' $R/src --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | grep -v '\.orig' | cut -c1-160
echo
echo '=== feature_layers / feature_rows 的所有出现 ==='
grep -rnE 'feature_layers|feature_rows' $R/src/targets --include=*.h 2>/dev/null | grep -v '\.orig' | head -12 | cut -c1-150
