#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== run_fp8 尾 12 ==="
tail -12 $J/dl/run_fp8.log 2>/dev/null
echo "=== 关键行 ==="
grep -E 'ACCEPTANCE=|PER_POS=|READY at|未就绪|FP8_RUN_DONE|服务退出' $J/dl/run_fp8.log 2>/dev/null | tail -6
echo "=== srv 尾 6 ==="
tail -6 $J/dl/srv_fp8.out 2>/dev/null
echo "=== 状态 ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1
