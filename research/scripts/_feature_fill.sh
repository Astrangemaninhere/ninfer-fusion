#!/bin/bash
# 追 features 的填充点（哪个层写入哪一段）——tap 层号的最终落点
R=/home/user/ninfer-fusion
echo "=== 1) feature_rows 的定义与使用 ==="
grep -rnE 'feature_rows' $R/src/targets/qwen3_6_27b/impl/config.h $R/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | grep -v '\.orig' | head -12
echo
echo "=== 2) 谁在写 features（capture/tap 点） ==="
grep -rnE 'features|feature_slice|tap_|capture_hidden|dflash_feature' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | head -20
echo
echo "=== 3) 运行时里按层写特征的代码（dflash_context / workspace_recipe） ==="
grep -rnE 'dflash_context|dflash_feature|feature_row|per_layer_feature|hidden_capture' $R/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | grep -v '\.orig' | head -20
echo
echo "=== 4) 契约文档里 tap 的定义 ==="
ls -1 $R/docs/maintainer/ 2>/dev/null | head -20
grep -rniE 'target_layer_ids|tap' $R/docs/maintainer/*.md 2>/dev/null | head -12
