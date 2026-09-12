#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== all ops::swa( call sites in tree:"
grep -rn 'ops::swa(\|ops::swa ' "$A/src" "$A/apps" "$A/tests" 2>/dev/null | head -20
echo
echo "=== all include of ninfer/ops/swa.h:"
grep -rn 'ninfer/ops/swa.h' "$A/src" "$A/include" "$A/apps" "$A/tests" 2>/dev/null | head
echo
echo "=== bidirectional_gqa include/call sites:"
grep -rn 'bidirectional_gqa_attention(' "$A/src" "$A/apps" "$A/tests" 2>/dev/null | grep -v launcher | head
echo
echo "=== swa usage in mxaim / main model runtime (grep 'swa' in targets):"
grep -rn 'swa' "$A/src/targets" --include=*.h --include=*.cpp | grep -v dflash | head -30
echo
echo "=== layouts_impl.h:600-660 (workspace recipe for swa/bidirectional):"
sed -n '600,660p' "$A/src/targets/qwen3_6/impl/runtime/layouts_impl.h" | cat -n
