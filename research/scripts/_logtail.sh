#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/train-dflash2.log
echo "=== 全文件里 mask/shift 痕迹 ==="
grep -inE 'mask|shift' "$L" | head -10
echo
echo "=== 日志总行数与最后一次启动的 banner（尾部 45 行） ==="
wc -l < "$L"
tail -45 "$L"
echo
echo "=== 编译 ==="
pgrep -x make >/dev/null && echo "make 在跑" || tail -8 /mnt/c/Users/User/Documents/ziqinzhang/dl/build_split.log
