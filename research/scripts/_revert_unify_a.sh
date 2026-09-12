#!/bin/bash
# 回退 UNIFY-A 的 373 行分派统一（未证实改动不留树上）→ 重编
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/unify_a_revert.log
export PATH=/home/user/.local/bin:$PATH
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== revert UNIFY-A $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3

echo "--- revert ---"
patch -p1 -R < "$J/_collab/UNIFY_A_patch.diff" > /tmp/rev_a.out 2>&1
rc=$?
echo "revert rc=$rc"
tail -3 /tmp/rev_a.out
[ "$rc" -ne 0 ] && { echo REVERT_FAIL; exit 2; }

echo "--- touch 被改头的 includer ---"
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
done < <(grep -E '^\+\+\+ ' "$J/_collab/UNIFY_A_patch.diff" | sed 's|^+++ b/||')

echo "--- build ---"
cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j2 2>&1 | tail -3
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
ls -l --time-style=+%H:%M ./apps/ninfer
echo REVERT_A_DONE
