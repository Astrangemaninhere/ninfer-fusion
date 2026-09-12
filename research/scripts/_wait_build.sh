#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_todo_129.py"
sleep 120
echo "--- 编译进度 ---"
tail -5 "$J/dl/rebuild_after_s35.log" | cut -c1-150
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do
  echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"
  break
done
echo "--- 二进制时间戳（未完成时仍是旧的） ---"
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
