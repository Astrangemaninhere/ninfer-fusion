#!/bin/bash
# 编入 KV dump 门放宽（Prefill|Verify），供 plain vs verify 的逐位置 KV 对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/kvdump_gate_build.log
export PATH=/home/user/.local/bin:$PATH
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== build kvdump gate $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3
for f in $(grep -rl 'text_context_impl.h' src --include=*.cpp --include=*.cu 2>/dev/null); do touch "$f"; done
echo "touched includers of text_context_impl.h"
cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j2 2>&1 | tail -4
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
ls -l --time-style=+%H:%M ./apps/ninfer
echo KVD_CATE_BUILD_DONE
