#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== run_fp8b 尾 12 ==="
tail -12 $J/dl/run_fp8b.log 2>/dev/null
echo "=== srv_fp8b 尾 8 ==="
tail -8 $J/dl/srv_fp8b.out 2>/dev/null
echo "=== 关键行 ==="
grep -E 'ACCEPTANCE=|PER_POS=|READY at|FP8B_DONE|服务退出' $J/dl/run_fp8b.log 2>/dev/null | tail -5
echo "=== 状态 ==="
pgrep -f 'run_fp8b' >/dev/null && echo "launcher 在" || echo "launcher 结束"
pgrep -f 'vllm.entrypoints' >/dev/null && echo "server 在跑" || echo "server 不在"
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1
