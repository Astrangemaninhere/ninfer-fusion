#!/bin/bash
# 窗口①：落 PART-B（FP32 partial_acc，29 文件超集）→ 编译 → 四项目基准 + 列0 一致率
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/land_part_b.log
export PATH=/home/user/.local/bin:$PATH
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== land PART-B (FP32 partial_acc) $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3

echo "--- dry-run ---"
patch -p1 --dry-run < "$J/_collab/PART_B_patch.diff" > /tmp/pb_dry.log 2>&1
rc=$?; echo "dry rc=$rc"; grep -cE 'FAILED|Hunk' /tmp/pb_dry.log
[ "$rc" -ne 0 ] && { echo DRY_FAIL; tail -8 /tmp/pb_dry.log; exit 2; }

echo "--- apply (-b) ---"
patch -p1 -b < "$J/_collab/PART_B_patch.diff" > /tmp/pb_apply.log 2>&1
rc=$?; echo "apply rc=$rc"
[ "$rc" -ne 0 ] && { echo APPLY_FAIL; tail -8 /tmp/pb_apply.log; exit 3; }

echo "--- touch 被改头的 includer（.cuh/.h 都要） ---"
grep -E '^\+\+\+ ' "$J/_collab/PART_B_patch.diff" | sed 's|^+++ b/||' > /tmp/pb_files.txt
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
done < /tmp/pb_files.txt
echo "files in patch: $(wc -l < /tmp/pb_files.txt)"

echo "--- 断言：live 核里不应再有 BF16 partial ---"
grep -n 'partial_acc' "$R/src/ops/kernel/gqa_attention_decode_bf16.cuh" | head -4

echo "--- build ---"
cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j2 2>&1 | tail -4
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }
ls -l --time-style=+%H:%M ./apps/ninfer

echo "--- 四项目基准 ---"
tr -d '\r' < "$J/_baseline_before_fix.sh" > /tmp/bbf4.sh
bash /tmp/bbf4.sh > /dev/null 2>&1
tail -8 "$J/dl/baseline_before_fix.log"
grep -E 'acceptance rate|accepted by pos|decode speed' /home/user/bl_zh_df2.log | tail -3
grep -E 'decode speed' /home/user/bl_zh_plain.log | tail -1

echo "--- 列 0 一致率（核心判据：69% → ?） ---"
tr -d '\r' < "$J/_col_agree_run.sh" > /tmp/car2.sh
bash /tmp/car2.sh 2>&1 | tail -12
echo LAND_PART_B_DONE
