#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== [a] verify 路径怎么算 target_tokens（有没有 policy） ==="
sed -n '355,395p' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | cat -n | sed 's/^/  355+/' | cut -c1-150
echo
echo "=== [b] policy 的实现（对哪些配置非恒等） ==="
sed -n '118,140p' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | cat -n | sed 's/^/  118+/' | cut -c1-150
echo
echo "=== [c] 本模型(qwen3.8-27b) 的 policy 相关常量 ==="
grep -nE 'final_logit_softcapping|output_multiplier|softcap|logit' "$R/src/targets/qwen3_6_27b/impl/config.h" | head -8 | cut -c1-130
echo
echo "=== [d] verify 里 argmax 的实际调用点（target_tokens 从哪来） ==="
grep -rn 'target_tokens' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | head -8 | cut -c1-150
