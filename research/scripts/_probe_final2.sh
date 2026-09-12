#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 编译状态 ==="
tail -6 "$J/dl/rebuild_after_s35.log" | cut -c1-150
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo
echo "=== 验证链（dspark 位置剖面） ==="
tail -20 "$J/dl/offset_family_verify.log" 2>/dev/null | cut -c1-180
pgrep -af '_offset_family_verify' | head -1 | cut -c1-55 || echo "  （验证链已结束）"
