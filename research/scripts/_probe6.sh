#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
M="$J/ninfer-fusion-repo/src/ops/launcher/gqa_attention_prefill.cu"
echo "=== memory ==="
free -g | head -3
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi n/a)"
echo
echo "=== prefill mirror 34-100 ==="
sed -n '34,100p' "$M" | cat -n | sed 's/^/  /'
echo "=== prefill mirror 110,125 ==="
sed -n '110,125p' "$M"
echo "=== prefill mirror 275,300 ==="
sed -n '275,300p' "$M"
echo "=== prefill mirror 340,365 ==="
sed -n '340,365p' "$M"
