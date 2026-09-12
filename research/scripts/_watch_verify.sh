#!/bin/bash
# 轮询等待编译产出新二进制（不 sleep 空等：每轮都检查真实条件），
# 一刷新就立刻跑验收脚本。
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
BIN=$R/build/apps/ninfer
LOG=$J/dl/watch_verify.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
baseline=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
echo "=== watch $(date '+%F %H:%M:%S') baseline=$baseline mtime=$(date -d @$baseline '+%H:%M') ==="
deadline=$(( $(date +%s) + 510 ))
fresh=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  now=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
  if [ "$now" -gt "$baseline" ]; then
    echo "binary refreshed: $(date -d @$now '+%H:%M')"
    fresh=1
    break
  fi
  if ! pgrep -f 'nvlink|nvcc|cc1plus|ptxas' >/dev/null 2>&1; then
    echo "no compiler processes and binary not refreshed (build failed?)"
    pgrep -af 'make ninfer' >/dev/null 2>&1 || echo "make is gone"
    break
  fi
  sleep 10
done
echo "exit: fresh=$fresh"
if [ "$fresh" -eq 1 ]; then
  echo "--- running verification ---"
  bash /tmp/verify.sh 2>&1 | tail -40
else
  echo "WATCH_TIMEOUT_OR_FAIL"
  tail -25 "$J/dl/build_df2head.log"
fi
echo WATCH_DONE
