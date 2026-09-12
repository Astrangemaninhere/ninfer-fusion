#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== srv_f8 尾 8 ==="
tail -8 $J/dl/srv_f8.out 2>/dev/null
echo "=== run_f8 尾 8 ==="
tail -8 $J/dl/run_f8.log 2>/dev/null
echo "=== 状态 ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
echo "=== 出数没 ==="
grep -E 'ACCEPTANCE=|PER_POS=' $J/dl/run_f8.log 2>/dev/null | tail -4
