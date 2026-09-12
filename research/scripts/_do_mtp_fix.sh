#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_fix_mtp_valid.py"
echo
echo "=== 编译状态 ==="
tail -4 "$J/dl/rebuild_after_s35.log" | cut -c1-140
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo
echo "=== 验证链 ==="
tail -14 "$J/dl/offset_family_verify.log" 2>/dev/null | cut -c1-175
pgrep -af '_offset_family_verify' | head -1 | cut -c1-55 || echo "  （已结束）"
