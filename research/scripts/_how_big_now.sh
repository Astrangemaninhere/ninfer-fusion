#!/bin/bash
R=/home/user/ninfer-fusion/build
echo "=== 当前正在编译的 TU（cmdline） ==="
pgrep -a -x nvcc | head -2 | sed 's/-forward-unknown-to-host-compiler//' | cut -c1-200
pgrep -a -x cicc | head -1 | cut -c1-120
pgrep -a -x ptxas | head -1 | cut -c1-160
echo
echo "=== 当前编译进程数与负载 ==="
echo "nvcc=$(pgrep -c -x nvcc 2>/dev/null) ptxas=$(pgrep -c -x ptxas 2>/dev/null) cicc=$(pgrep -c -x cicc 2>/dev/null)"
uptime | sed 's/.*load average/load/'
echo
echo "=== 最大的 8 个 .o（重编代价的来源） ==="
find "$R" -name '*.o' -printf '%s %p\n' 2>/dev/null | sort -rn | head -8 | \
  awk '{printf "  %8.1f MB  %s\n", $1/1048576, $2}'
echo
echo "=== 正在写的临时 .o（说明在编哪个） ==="
find "$R" -name '*.o' -newermt '-15 minutes' -printf '%s %p\n' 2>/dev/null | sort -rn | head -4 | \
  awk '{printf "  %8.1f MB  %s\n", $1/1048576, $2}'
echo
echo "=== 二进制/日志 ==="
ls -l --time-style=+%H:%M "$R/apps/ninfer" 2>/dev/null
tail -2 /mnt/c/Users/User/Documents/ziqinzhang/dl/kvdump_gate_build.log
