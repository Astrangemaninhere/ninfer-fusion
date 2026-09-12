#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 日志尾 ==="
tail -6 "$J/dl/kvdump_gate_build.log" 2>/dev/null
echo
echo "=== 进程/负载 ==="
echo "make=$(pgrep -c -x make 2>/dev/null) nvcc=$(pgrep -c -x nvcc 2>/dev/null) nvlink=$(pgrep -c -x nvlink 2>/dev/null) ptxas=$(pgrep -c -x ptxas 2>/dev/null) cicc=$(pgrep -c -x cicc 2>/dev/null)"
uptime | sed 's/.*load average/load/'
free -g | head -2
echo
echo "=== 正在跑什么（cmdline 摘要） ==="
pgrep -a -x nvcc | head -1 | cut -c1-160
pgrep -a -x nvlink | head -1 | cut -c1-120
ps -o pid,etime,rss,pcpu,comm -C nvcc,nvlink,make 2>/dev/null | head -6
echo
echo "=== 产物时间 ==="
ls -l --time-style=+%F_%H:%M "$R/build/apps/ninfer" 2>/dev/null
ls -l --time-style=+%H:%M "$R/build/src/CMakeFiles/ninfer_ops.dir/cmake_device_link.o" 2>/dev/null
