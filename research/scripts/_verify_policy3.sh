#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== policy 的常量名与判定 ==="
sed -n '100,130p' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | cat -n | sed 's/^/  100+/' | cut -c1-150
echo
echo "=== 这些常量在 27b/家族 配置里的值 ==="
grep -rnE 'softcap|multiplier|final_logit' "$R/src/targets/qwen3_6_27b/impl/config.h" "$R/src/targets/qwen3_6/impl/config.h" 2>/dev/null | head -10 | cut -c1-150
echo
echo "=== apply_final_logit_policy 的真实实现（含判空/恒等分支） ==="
grep -n -A16 'static void apply_final_logit_policy' "$R/src/targets/qwen3_6/impl/runtime/text_context.h" | head -24 | cut -c1-150
