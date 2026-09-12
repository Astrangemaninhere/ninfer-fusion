#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 基线脚本里 plain 的落点与 ids 提取 ==="
grep -nE 'plain|generated ids|BL_|bl_.*log|token|greedy' $J/_baseline_before_fix.sh | head -30
echo
echo "=== dl/ 下今天 17:5x-18:0x 的日志 ==="
ls -lt --time-style=+%H:%M $J/dl/*.log 2>/dev/null | head -12
