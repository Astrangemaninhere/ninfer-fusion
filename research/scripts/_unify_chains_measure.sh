#!/bin/bash
# 单变量：只含 FP8 累加链 4->1（统一算术的最小解）的编译 + 四项目基准
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/unify_chains.log
export PATH=/home/user/.local/bin:$PATH
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== unify chains-only: build + measure $(date '+%F %H:%M:%S') ==="
cd "$R/build" || exit 3
start=$(date +%s)
make ninfer -j2 2>&1 | tail -4
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 1; }
ls -l --time-style=+%H:%M ./apps/ninfer

echo "--- 四项目基准 ---"
bash /tmp/bbf3.sh > /dev/null 2>&1 || bash /tmp/bbf.sh > /dev/null 2>&1 || echo "基准脚本缺失"
tail -8 "$J/dl/baseline_before_fix.log"
echo "--- 接受率/速度 ---"
grep -E 'acceptance rate|accepted by pos|decode speed' /home/user/bl_zh_df2.log | tail -3
echo "--- T=1 侧（plain）速度代价 ---"
grep -E 'decode speed' /home/user/bl_zh_plain.log | tail -1
echo UNIFY_CHAINS_DONE
