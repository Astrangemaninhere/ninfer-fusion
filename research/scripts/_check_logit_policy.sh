#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== apply_final_logit_policy 的调用点 ==="
grep -rn 'apply_final_logit_policy' "$R/src" 2>/dev/null | cut -c1-155
echo
echo "=== verify 路径：target_verify_batch 的定义与 argmax ==="
grep -rn 'target_verify_batch' "$R/src" 2>/dev/null | head -5 | cut -c1-140
echo
echo "=== qwen3_6 runtime 里 argmax 的使用（plain 解码 vs verify） ==="
grep -rn 'ops::argmax\|argmax(' "$R/src/targets/qwen3_6/impl/runtime/"*.h 2>/dev/null | head -10 | cut -c1-150
