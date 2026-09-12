#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 编译状态 ==="
tail -6 "$J/dl/rebuild_after_s35.log" | cut -c1-150
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo
echo "=== 对照链状态 ==="
tail -20 "$J/dl/post_fix_verify.log" 2>/dev/null | cut -c1-180
pgrep -af '_post_fix_verify' | head -1 | cut -c1-55 || echo "  （对照链已结束）"
