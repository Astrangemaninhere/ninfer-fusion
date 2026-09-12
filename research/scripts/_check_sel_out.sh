#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang/dl
echo "--- stdout df2sel 计数 ---"
grep -c df2sel "$D/scale_probe_stdout.log" 2>&1
echo "--- 前 12 行 ---"
grep df2sel "$D/scale_probe_stdout.log" 2>&1 | head -12
echo "--- stdout 行数 ---"
wc -l "$D/scale_probe_stdout.log" 2>&1
echo "--- stderr 里有没有别的线索 ---"
tail -5 "$D/scale_probe_stderr.log" 2>&1
echo "--- 文件列表 ---"
ls -l "$D"/scale_probe_*.log 2>&1
