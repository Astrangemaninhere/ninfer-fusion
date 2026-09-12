#!/usr/bin/env bash
# ninfer 快速分层构建: ops 先行(多数编译错误在此), 再 engine, 最后 serve 链接.
# 用法: ./build_fast.sh [target...]   (默认 ninfer-serve)
# 内存纪律: 14GB VM 下 cicc 峰值 ~4GB -> -j8 为安全并发(swap 兜底); 勿用 -j32.
set -u
cd /home/user/ninfer-fusion/build || exit 2
JOBS=${JOBS:-8}
TARGET=${1:-ninfer-serve}
LOG=/tmp/ninfer_build.log
echo "[build] jobs=$JOBS target=$TARGET -> $LOG"
for t in ninfer_core ninfer_ops ninfer_engine "$TARGET"; do
  echo "[build] == $t =="
  if ! cmake --build . --target "$t" -j"$JOBS" >> "$LOG" 2>&1; then
    echo "[build] FAILED at $t (tail):"
    tail -25 "$LOG"
    exit 1
  fi
done
echo "[build] OK $(tail -1 "$LOG")"
