#!/bin/bash
F=/home/user/ninfer-fusion/src/ops/launcher/gqa_attention_decode.cu
echo '=== 1..40 (API/包含) ==='
awk 'NR<=40 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-115
echo
echo '=== 560..737 (派发与实例化) ==='
awk 'NR>=560 && NR<=737 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-115
