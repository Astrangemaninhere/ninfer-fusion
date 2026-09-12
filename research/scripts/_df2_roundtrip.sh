#!/bin/bash
# 重跑 roundtrip（用 Windows 风格路径），并汇报编译现场
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
PY="/mnt/c/Program Files/Python312/python.exe"
WJ="C:/Users/User/Documents/ziqinzhang"
LOG=$J/dl/df2_roundtrip.log
cd "$J" || exit 3
exec > >(tee -a "$LOG") 2>&1
echo "=== roundtrip 闸门 $(date '+%F %H:%M:%S') ==="
for CK in step_000200 step_001900; do
  echo "########## $CK ##########"
  "$PY" "$WJ/_dflash2_roundtrip_tmp.py" "$WJ/data/dflash2_ckpts/$CK.pt" \
        "$WJ/data/dflash2_ckpts/${CK}_tuned.ninfer" 2>&1 | tail -6
done
echo
echo "=== 编译现场 ==="
ps -eo etimes,args | grep 'make ninfer' | grep -v grep | cut -c1-14
ps -eo etimes,time,pcpu,rss,comm --sort=-time | head -4
ps -eo args | grep -oE 'src/[a-zA-Z0-9_/]+\.cu' | sort -u | head -4
free -g | head -2
tail -3 /mnt/c/Users/User/Documents/ziqinzhang/dl/build_split.log
echo DF2_ROUNDTRIP_DONE
