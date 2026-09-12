#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== $(date +%H:%M:%S) ==="
echo "=== prob 变体（8128）日志尾 ==="
tail -n 8 "$J/dl/prob_run.log" 2>/dev/null | cut -c1-120
echo
echo "=== A5b A/B 重编 ==="
tail -n 6 "$J/dl/ab_a5b.log" 2>/dev/null | cut -c1-120
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)"
echo
echo "=== 两个二进制是否已生成 ==="
ls -l --time-style=+%H:%M /home/user/ninfer_no_a5b /home/user/ninfer_with_a5b 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
