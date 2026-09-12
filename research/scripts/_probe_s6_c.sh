#!/usr/bin/env bash
# S6 recon probe C: per-TU wall time from par_build.log timestamps (read-only)
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/par_build.log

echo "=== log: make invocations / totals ==="
grep -n '^--- make\|^=== par_build\|^--- 浜' "$LOG" | tail -12

echo
echo "=== log: all gqa_attention lines (with timestamps) ==="
grep -n 'gqa_attention' "$LOG"

echo
echo "=== log: first 5 and last 25 lines ==="
head -5 "$LOG"
echo "   ..."
tail -25 "$LOG"

echo
echo "=== current time / running nvcc ==="
date '+%F %H:%M:%S'
ps -eo pid,etimes,comm,args --no-headers | grep -E 'nvcc|cc1plus|cicc|ptxas' | grep -v grep | head -10

echo
echo "=== object mtimes (H:M) ==="
ls -l --time-style=+%H:%M:%S /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention*.o
