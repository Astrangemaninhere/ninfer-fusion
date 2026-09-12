#!/bin/bash
# 重编 A5b 已回退的树（CPU 侧；与 GPU 上的 vLLM 跑不冲突）
set -u
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'dflash_impl.h' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
N=$(printf '%s\n' "$T" | grep -c . || true)
echo "touch $N 个包含者"
[ -n "$T" ] && touch $T
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve >/dev/null 2>&1 &
sleep 15
echo "nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)"
tail -2 /mnt/c/Users/User/Documents/ziqinzhang/dl/par_build.log | cut -c1-110
