#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== mem_trace（每 2 秒：时间 内存GB 峰值GB 显存GB 当前阶段） ==="
tail -25 $J/dl/mem_trace.log 2>/dev/null
echo
echo "=== cap_outer 尾 8 ==="
tail -8 $J/dl/cap_outer.log 2>/dev/null
echo
echo "=== srv_cap 尾 8 ==="
tail -8 $J/dl/srv_cap.out 2>/dev/null
echo
echo "=== 宿主看门狗尾 4 ==="
tail -4 $J/dl/watchdog2.log 2>/dev/null
echo "=== VM 内 free ==="
free -g | head -2
