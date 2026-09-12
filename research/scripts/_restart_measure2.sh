#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_fix_measure_script2.py" || exit 1
bash -n "$J/_post_build_measure.sh" && echo "SYNTAX_OK" || exit 1
grep -n 'no-thinking' "$J/_post_build_measure.sh" | head -3 | cut -c1-120
grep -n 'INVALID - empty text' "$J/_post_build_measure.sh" | head -2 | cut -c1-120
echo
rm -f /tmp/pb_rows /tmp/pb_verdict.txt /home/user/pb_verdict.txt /tmp/pb_text_*.txt
: > /tmp/pb_rows
echo "=== 重启测量 ==="
setsid nohup bash "$J/_post_build_measure.sh" > /dev/null 2>&1 < /dev/null &
sleep 20
tail -5 "$J/dl/postbuild_measure.log" | cut -c1-150
