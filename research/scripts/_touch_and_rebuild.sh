#!/bin/bash
# 补 touch：config.h 的传递包含者（variant.cpp / package.cpp 等），然后真重编
set -u
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
cd "$R" || exit 1

echo '=== 直接 include config.h 的源文件 ==='
D=$(grep -rln --include='*.cpp' --include='*.cu' 'impl/config.h' src/targets/qwen3_6_27b/ 2>/dev/null | sort -u)
echo "$D" | sed 's/^/  /'
echo '=== 直接 include dflash_impl.h / package.h 的源文件 ==='
E=$(grep -rln --include='*.cpp' --include='*.cu' -e 'dflash_impl.h' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
echo "$E" | sed 's/^/  /'
ALL=$(echo -e "$D\n$E" | sort -u | grep -v '^$')
echo "共 touch $(echo "$ALL" | wc -l) 个"
[ -n "$ALL" ] && touch $ALL

echo
echo '=== 重编 ==='
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve 2>&1 | tail -8
echo "  rc=$?"
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" | awk '{print "  ninfer:", $6, $5}'
date +%H:%M:%S
