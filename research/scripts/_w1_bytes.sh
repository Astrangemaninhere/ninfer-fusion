#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== kernel 106-115 (template+signature):"
sed -n '106,115p' "$A/src/ops/kernel/bidirectional_gqa_attention.cuh" | cat -A | sed -e 's/\$$/<EOL>/'
echo
echo "=== kernel 362-392 (allow block) raw with EOL marks:"
sed -n '362,392p' "$A/src/ops/kernel/bidirectional_gqa_attention.cuh" | cat -A | sed -e 's/\$$/<EOL>/'
echo
echo "=== kernel 536-549 (swa kernel template):"
sed -n '536,549p' "$A/src/ops/kernel/bidirectional_gqa_attention.cuh" | cat -A | sed -e 's/\$$/<EOL>/'
