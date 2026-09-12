#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_offset_family_verify.sh" && echo SYNTAX_OK || exit 1
setsid nohup bash "$J/_offset_family_verify.sh" > /dev/null 2>&1 < /dev/null &
sleep 6
pgrep -af '_offset_family_verify' | cut -c1-70
echo "--- 编译进度 ---"
tail -3 "$J/dl/rebuild_after_s35.log" | cut -c1-130
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
