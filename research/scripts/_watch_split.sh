#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/build_split.log
for i in $(seq 1 640); do
  if grep -qE 'BUILD_SPLIT_OK|BUILD_SPLIT_FAIL' "$L" 2>/dev/null; then break; fi
  sleep 5
done
echo "=== 拆分编译收尾 $(date '+%H:%M:%S') ==="
sed -n '/--- 7)/,$p' "$L" | tail -25
