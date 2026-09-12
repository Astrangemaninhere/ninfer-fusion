#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== run_fi 尾 12 ==="
tail -12 $J/dl/run_fi.log 2>/dev/null
echo "=== 关键行 ==="
grep -E 'READY at|ACCEPTANCE=|PER_POS=|FI_RUN_DONE|未就绪|服务退出' $J/dl/run_fi.log 2>/dev/null | tail -6
echo "=== srv_fi 尾 6 ==="
tail -6 $J/dl/srv_fi.out 2>/dev/null
echo "=== VM 内存 ==="
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1
echo "=== 宿主看门狗（最近 6 行） ==="
tail -6 $J/dl/watchdog.log 2>/dev/null
