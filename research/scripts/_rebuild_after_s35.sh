#!/bin/bash
# 重建：make ninfer（含 engine.cpp + 3 个 variant TU）→ make ninfer-serve（链接实测用二进制）
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/rebuild_after_s35.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== rebuild after S35 compile fix $(date '+%F %H:%M:%S') ==="
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
# ccache lives in ~/.local/bin and the build launcher is `ccache nvcc`:
# without this the make dies with "ccache: not found" (Error 127).
export PATH="/home/user/.local/bin:$PATH"
cd $R/build || exit 3
for target in ninfer ninfer-serve; do
  rc=1
  for attempt in 1 2 3; do
    echo "--- $target attempt $attempt $(date +%H:%M:%S) ---"
    free -g | sed -n 2p
    make "$target" -j1 > /tmp/reb_${target}_$attempt.log 2>&1
    rc=$?
    tail -3 /tmp/reb_${target}_$attempt.log
    [ $rc -eq 0 ] && break
    echo "  errors:"
    grep -E 'error:' /tmp/reb_${target}_$attempt.log | head -5 | cut -c1-170
    sleep 5
  done
  echo "  $target rc=$rc"
done
echo "--- binaries ---"
ls -l --time-style=+%m-%d_%H:%M $R/build/apps/ninfer $R/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
ls -l --time-style=+%H:%M $R/build/src/CMakeFiles/ninfer_engine.dir/targets/*/impl/variant.cpp.o 2>/dev/null | awk '{print "  variant", $6, $5, $NF}'
echo "=== rebuild done $(date '+%F %H:%M:%S') ==="
