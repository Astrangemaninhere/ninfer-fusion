#!/bin/bash
# 叠加 UNIFY-A 的 373 行分派统一（在 FP8 累加链改动之上）→ 编译 → 四项目基准
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/unify_a_apply.log
export PATH=/home/user/.local/bin:$PATH
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== apply UNIFY-A + measure $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3

echo "--- dry-run ---"
patch -p1 --dry-run < "$J/_collab/UNIFY_A_patch.diff" > /tmp/ua_dry.log 2>&1
rc=$?; echo "dry rc=$rc"; tail -3 /tmp/ua_dry.log
[ "$rc" -ne 0 ] && { echo DRY_FAIL; exit 2; }

echo "--- apply (-b) ---"
patch -p1 -b < "$J/_collab/UNIFY_A_patch.diff" > /tmp/ua_apply.log 2>&1
rc=$?; echo "apply rc=$rc"
[ "$rc" -ne 0 ] && { echo APPLY_FAIL; tail -5 /tmp/ua_apply.log; exit 3; }

echo "--- touch 被改头的 includer ---"
grep -E '^\+\+\+ ' "$J/_collab/UNIFY_A_patch.diff" | sed 's|^+++ b/||' > /tmp/ua_files.txt
while read -r f; do
  [ -f "$f" ] || continue
  case "$f" in
    *.cuh|*.h)
      b=$(basename "$f")
      for inc in $(grep -rl "$b" "$R/src" 2>/dev/null | sed "s|$R/||"); do
        case "$inc" in *.cuh|*.h) continue;; esac
        touch "$R/$inc"
      done
      ;;
  esac
done < /tmp/ua_files.txt
echo "touched: $(wc -l < /tmp/ua_files.txt) 个文件（含头者展开）"

echo "--- build ---"
cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j2 2>&1 | tail -4
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }
ls -l --time-style=+%H:%M ./apps/ninfer

echo "--- 四项目基准（含 T=1 代价） ---"
bash /tmp/bbf3.sh > /dev/null 2>&1 || bash /tmp/bbf.sh > /dev/null 2>&1 || echo "基准脚本缺失"
tail -8 "$J/dl/baseline_before_fix.log"
grep -E 'acceptance rate|accepted by pos|decode speed' /home/user/bl_zh_df2.log | tail -3
echo "--- plain(T=1) 代价 ---"
grep -E 'decode speed' /home/user/bl_zh_plain.log | tail -1
echo UNIFY_A_APPLIED_DONE
