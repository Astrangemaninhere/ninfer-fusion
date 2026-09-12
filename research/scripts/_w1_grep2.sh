#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== who calls dflash2_* in program_impl.h:"
grep -n 'dflash2' "$A/src/targets/qwen3_6/impl/runtime/program_impl.h" | head -40
echo
echo "=== 'acceptance rate' / 'accepted by pos' emitters:"
grep -rn 'acceptance rate\|accepted by pos\|accepted_by_position\|accepted by position' "$A/src" "$A/apps" | head -20
echo
echo "=== swa wrapper full:"
cat -n "$A/src/ops/wrapper/swa.cpp"
