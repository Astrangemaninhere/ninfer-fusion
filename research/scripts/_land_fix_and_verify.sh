#!/bin/bash
# 落地任一补丁并跑同一套回归（P2 纪律：备份 + dry-run + -b 应用 + 可 patch -R 回滚）。
# 用法: bash /tmp/land.sh <patch 文件绝对路径> <tag>
set -u
PATCH=${1:?需要补丁路径}
TAG=${2:-fix}
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/land_$TAG.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== land $TAG = $(basename "$PATCH") $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3

echo "--- dry-run ---"
patch -p1 --dry-run < "$PATCH"; rc=$?
echo "dry-run rc=$rc"
[ "$rc" -ne 0 ] && { echo DRY_RUN_FAIL; exit 2; }

echo "--- 将被改动的文件与 md5（改前） ---"
grep -E '^\+\+\+ ' "$PATCH" | sed 's|^+++ b/||' > /tmp/land_files.txt
while read -r f; do [ -f "$f" ] && echo "  $(md5sum "$f" | cut -c1-12)  $f"; done < /tmp/land_files.txt

echo "--- apply (-b 备份) ---"
patch -p1 -b < "$PATCH"; rc=$?
echo "apply rc=$rc"
[ "$rc" -ne 0 ] && { echo APPLY_FAIL; exit 3; }
echo "--- 改后 md5 ---"
while read -r f; do [ -f "$f" ] && echo "  $(md5sum "$f" | cut -c1-12)  $f"; done < /tmp/land_files.txt

echo "--- touch includers（无头文件依赖跟踪） ---"
while read -r f; do
  base=$(basename "$f")
  for inc in $(grep -rl "$base" "$R/src" 2>/dev/null | sed "s|$R/||"); do
    case "$inc" in *.cuh|*.h) continue;; esac
    touch "$R/$inc"; echo "  touched $inc"
  done
done < /tmp/land_files.txt

echo "--- build ---"
cd "$R/build" || exit 4
make ninfer -j2 2>&1 | tail -6
rc=${PIPESTATUS[0]}
echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }
ls -l --time-style=+%H:%M ./apps/ninfer

echo "--- 回归（同一套基准脚本，会追加到 dl/baseline_before_fix.log） ---"
bash /tmp/bbf.sh > /dev/null 2>&1
tail -8 "$J/dl/baseline_before_fix.log"
echo LAND_${TAG}_DONE
