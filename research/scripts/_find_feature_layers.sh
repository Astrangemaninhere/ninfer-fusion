#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== 抓取上下文特征的入口 enqueue_dflash_context_append ==='
grep -rn 'enqueue_dflash_context_append' $R/src --include=*.h --include=*.cpp 2>/dev/null | grep -v '\.orig' | head -6 | cut -c1-150
echo
N=$(grep -n 'void ProgramImplCore::enqueue_dflash_context_append' $R/src/targets/qwen3_6/impl/runtime/program_impl.h | head -1 | cut -d: -f1)
if [ -n "$N" ]; then
  echo "=== 定义处 附近（$N）==="
  awk -v s=$N -v e=$((N+45)) 'NR>=s && NR<=e {printf "%5d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | cut -c1-155
fi
echo
echo '=== 特征层选择：feature_rows / 层索引常量 ==='
grep -rnE 'feature_rows|kDflashFeatureRows|feature_layer|context_layer_ids|target_layer_ids|kDflashLayers' $R/src/targets/qwen3_6/impl/config.h $R/src/targets/qwen3_6_27b/impl/config.h 2>/dev/null | head -12 | cut -c1-150
echo
echo '=== 预填特征收集点（哪一层写进 features）==='
grep -rnE 'prefill_features|pending_features|context_features' $R/src/targets/qwen3_6/impl/runtime/program_impl.h 2>/dev/null | head -12 | cut -c1-155
