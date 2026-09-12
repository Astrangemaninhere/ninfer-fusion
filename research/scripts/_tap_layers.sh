#!/bin/bash
# tap 层号取证：引擎从哪几层取隐藏态喂给 fc（与契约 [5,19,33,47,61] 对照）
R=/home/user/ninfer-fusion
echo "=== 1) target_layer_ids / tap 相关出现点 ==="
grep -rnE 'target_layer_ids|target_layers|tap|feature_projection|fc\.|draft_feature' $R/src/targets/qwen3_6_27b/impl $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | grep -v '\.orig' | head -25
echo
echo "=== 2) dflash2_impl.h 里构造特征的那段（fc 的输入） ==="
grep -n -B 6 -A 14 'feature_projection' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | head -50
echo
echo "=== 3) variant/config 里是否声明了 tap 层 ==="
grep -rnE 'target_layer|tap_layer|full_attention_layers|layer_ids' $R/src/targets/qwen3_6_27b/impl/config.h $R/src/targets/qwen3_6_27b/impl/variant.h $R/src/targets/qwen3_6/impl/runtime/layouts.h 2>/dev/null | head -15
