#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== dl/ 下训练相关日志（按时间） ==="
ls -lt --time-style=+%m-%d_%H:%M $J/dl/ 2>/dev/null | grep -iE 'train|df2|dflash' | head -10
echo
echo "=== 根目录 .py/.bat 训练入口 ==="
ls -lt --time-style=+%m-%d_%H:%M $J/train_dflash2.py $J/_train_df2*.bat 2>/dev/null | head
echo
echo "=== 09-10 22:00 之后被改过的日志/脚本（找新那炉的痕迹） ==="
find $J -maxdepth 2 -newermt "2026-09-10 22:00" -type f \( -name '*.log' -o -name '*.bat' -o -name '*.sh' -o -name '*.py' \) 2>/dev/null | head -12
echo
echo "=== 编译 ==="
pgrep -x make >/dev/null && echo "make 在跑 $(ps -eo etimes,args | grep 'make ninfer' | grep -v grep | head -1 | awk '{print $1"s"}')" || { echo 'make 已结束'; tail -6 /mnt/c/Users/User/Documents/ziqinzhang/dl/build_split.log; }
