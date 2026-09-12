#!/bin/bash
# 最后一步取证：谁填特征缓冲，tap 层号从哪来
R=/home/user/ninfer-fusion
echo "=== 1) feature_layers 定义 ==="
grep -rnE 'feature_layers' $R/src/targets/qwen3_6_27b/impl/config.h $R/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | grep -v '\.orig' | head -10
echo
echo "=== 2) 谁写 prefill_features / pending_features ==="
grep -rnE 'prefill_features|pending_features' $R/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | grep -v '\.orig' | head -20
echo
echo "=== 3) 文本侧把层输出写进特征的地方（找层号条件） ==="
grep -rnE 'feature|tap' $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h 2>/dev/null | head -25
echo
echo "=== 4) 是否出现契约里的 5 个层号 5/19/33/47/61（或 1-based 6/20/34/48/62） ==="
grep -rnE '\b(5|19|33|47|61|6|20|34|48|62)\b' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | grep -vE '//' | head -10
grep -rn 'target_layer' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -10
