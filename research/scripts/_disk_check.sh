#!/bin/bash
echo '=== 磁盘 ==='
df -h | grep -E 'Filesystem|/dev/sd|/mnt/c|/mnt/g' | head -8
echo
echo '=== WSL 交换文件 ==='
ls -la /mnt/c/Users/User/AppData/Local/Temp/*.vhdx 2>/dev/null | awk '{printf "  %.1f GB  %s\n", $5/1073741824, $NF}'
ls -la /mnt/c/Users/User/*.vhdx 2>/dev/null | awk '{printf "  %.1f GB  %s\n", $5/1073741824, $NF}'
echo
echo '=== 内存 / 交换 ==='
free -g
awk '/SwapTotal|SwapFree/{printf "  %s = %.1f GB\n", $1, $2/1048576}' /proc/meminfo
echo
echo '=== ninfer 构建 ==='
echo "  nvcc = $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
echo "  进度 = $(grep -oE '\[[ 0-9]+%\]' /mnt/c/Users/User/Documents/ziqinzhang/dl/par_build.log 2>/dev/null | tail -1)"
echo "  MemAvailable = $(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)"
echo "  二进制 = $(stat -c '%y' /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | cut -c1-19)"
echo "  最近构建日志行:"
tail -3 /mnt/c/Users/User/Documents/ziqinzhang/dl/build_spill_ssd.log 2>/dev/null | cut -c1-120 | sed 's/^/    /'
