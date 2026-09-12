#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
date +%H:%M:%S
echo '=== 批次日志尾 ==='
tail -22 "$J/dl/land_batch1.log" | cut -c1-130
echo '=== 构建 ==='
echo "nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1
echo "错误条数: $(grep -cE ' error:' "$J/dl/par_build.log" 2>/dev/null)"
ls -la --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | cut -c25-75
echo '=== 内存 ==='
free -g | sed -n 2p
