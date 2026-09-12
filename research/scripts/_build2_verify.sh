#!/bin/bash
# 补齐 35b / muse 的 DFlash2Config::draft_head_rows 后增量重编 + 验收。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/build_df2head2.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== build df2head2 $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
start=$(date +%s)
make ninfer -j2 2>&1 | tail -25
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "make rc=$rc elapsed=$(( (end-start)/60 ))m$(( (end-start)%60 ))s"
ls -l --time-style=+%H:%M $R/build/apps/ninfer
if [ "$rc" -ne 0 ]; then echo "BUILD_DF2HEAD2_FAIL"; exit 1; fi
echo "BUILD_DF2HEAD2_OK"
echo "--- verification ---"
bash /tmp/verify.sh 2>&1 | tail -35
echo BUILD2_AND_VERIFY_DONE
