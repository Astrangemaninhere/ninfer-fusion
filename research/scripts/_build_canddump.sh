#!/bin/bash
# 编译带候选 dump 的探针（只 touch 那一个包含者）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
LOG=$J/dl/build_canddump.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 编译 $(date '+%F %H:%M:%S') ==="
touch $R/src/ops/launcher/dflash2_selector.cu
cd "$R/build" || exit 3
start=$(date +%s)
make ninfer -j3 2>&1 | tail -8
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 4; }
ls -l --time-style=+%H:%M $R/build/apps/ninfer | awk '{print "  bin:", $5, $6}'
echo BUILD_CANDDUMP_OK
