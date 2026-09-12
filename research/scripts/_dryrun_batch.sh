#!/bin/bash
# 对当前树 dry-run 所有待落补丁，确认没有和已落地的 offset 修复冲突。
# 用法：bash /tmp/dryrun_batch.sh
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang/_collab
cd "$R" || exit 3

for f in "$J"/E9_s52_dflash2_k_slice.diff \
         "$J"/E7_s50_regression_sketch.diff \
         "$J"/A5b_attention_valid_width.diff \
         "$J"/A5_round_probe.diff \
         "$J"/A5_block_rows.diff; do
  name=$(basename "$f")
  [ -f "$f" ] || { echo "[MISSING] $name"; continue; }
  t="/tmp/d_$name"
  tr -d '\r' < "$f" > "$t"
  echo "=== $name ==="
  out=$(patch -p1 --dry-run < "$t" 2>&1)
  rc=$?
  echo "$out" | tail -8
  echo "  rc=$rc"
  echo
done
echo '=== _collab 里现有补丁类文件 ==='
ls -1 "$J" | grep -iE 'diff|patch|sketch' | head -20
