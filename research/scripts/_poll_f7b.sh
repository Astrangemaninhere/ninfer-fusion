#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== run_f7 尾 14 ==="
tail -14 $J/dl/run_f7.log 2>/dev/null
echo "=== 出数 ==="
grep -E 'ACCEPTANCE=|PER_POS=|完成 |请求失败|ready at|未就绪|F7_DONE' $J/dl/run_f7.log 2>/dev/null | tail -8
echo "=== srv_f7 尾 5 ==="
tail -5 $J/dl/srv_f7.out 2>/dev/null
echo "=== 状态 ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
