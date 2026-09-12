#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
sleep 260
echo "=== run_fp8 ==="
tail -14 $J/dl/run_fp8.log 2>/dev/null
echo "=== 出数 ==="
grep -E 'ACCEPTANCE=|PER_POS=|完成 |请求失败|READY at|未就绪|FP8_RUN_DONE' $J/dl/run_fp8.log 2>/dev/null | tail -6
echo "=== 状态 ==="
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | head -1
