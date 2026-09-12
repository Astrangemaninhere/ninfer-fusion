#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== verify 的 target_tokens 产生点（实现体） ==="
grep -n 'target_verify_batch_impl' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | head -3
L=$(grep -n 'void TextContext::target_verify_batch_impl' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | head -1 | cut -d: -f1)
echo "  实现起始行: $L"
if [ -n "$L" ]; then
  sed -n "${L},$((L+60))p" "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | grep -nE 'argmax|policy|logits|target_tokens|softcap|multiplier' | cut -c1-150
fi
echo
echo "=== 27b 的 policy 常量（[c] 重跑） ==="
grep -nE 'final_logit_softcapping|output_multiplier|softcap' "$R/src/targets/qwen3_6_27b/impl/config.h" | head -6 | cut -c1-140
echo "  --- 共享家族配置里呢 ---"
grep -rnE 'final_logit_softcapping|output_multiplier' "$R/src/targets/qwen3_6/impl/config.h" 2>/dev/null | head -6 | cut -c1-140
echo
echo "=== softcap_impl 的判定（什么情况下非恒等） ==="
grep -n -A8 'struct softcap_impl' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | head -20 | cut -c1-140
