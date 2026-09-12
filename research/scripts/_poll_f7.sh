#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
sleep 240
echo "=== run_f7 ==="
tail -16 $J/dl/run_f7.log 2>/dev/null
echo "=== 出数没 ==="
grep -E 'ACCEPTANCE=|PER_POS=|完成 |请求失败|ready at|未就绪' $J/dl/run_f7.log 2>/dev/null | tail -6
echo "=== 进程/内存/GPU ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
