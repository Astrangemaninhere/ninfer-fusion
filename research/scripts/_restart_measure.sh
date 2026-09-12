#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_fix_measure_script.py" || exit 1
bash -n "$J/_post_build_measure.sh" && echo "SYNTAX_OK" || exit 1
echo
echo "=== 清掉上一轮的半成品结果（避免与新一轮混在一起） ==="
rm -f /tmp/pb_rows /tmp/pb_verdict.txt /home/user/pb_verdict.txt
: > /tmp/pb_rows
echo "  已清"
echo
echo "=== 重启测量（detached） ==="
setsid nohup bash "$J/_post_build_measure.sh" > /dev/null 2>&1 < /dev/null &
sleep 15
tail -6 "$J/dl/postbuild_measure.log" | cut -c1-150
pgrep -af '_post_build_measure.sh' | cut -c1-70
