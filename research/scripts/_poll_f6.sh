#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== run_f6 日志 ==="
tail -20 $J/dl/run_f6.log 2>/dev/null
echo
echo "=== 是否出数 ==="
grep -E 'ACCEPTANCE=|PER_POS=|完成 |请求失败|ready at' $J/dl/run_f6.log 2>/dev/null | tail -6
echo
echo "=== 进程/内存/GPU ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
echo
echo "=== 服务日志尾 ==="
tail -6 $J/dl/srv_f6.out 2>/dev/null
