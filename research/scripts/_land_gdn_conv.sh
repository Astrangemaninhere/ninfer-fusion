#!/bin/bash
# 落地 UP1_gdn_conv_column.diff：GDN 短卷积滚动窗口应回带"已发布的 BF16 列"，而不是更宽的累加器。
set -u
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/gdn_conv_land.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== land UP1_gdn_conv_column $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3

F=src/ops/gdn_input_proj/gdn_conv.cuh
mkdir -p /home/user/gdnconv_bak
cp -p "$F" /home/user/gdnconv_bak/gdn_conv.cuh.$(date +%H%M%S)
echo "备份: $(ls -t /home/user/gdnconv_bak/ | head -1)  md5_before=$(md5sum "$F" | cut -c1-12)"

echo "--- dry-run ---"
patch -p1 --dry-run < "$D/UP1_gdn_conv_column.diff"
echo "dry-run rc=$?"

echo "--- apply (with -b) ---"
patch -p1 -b < "$D/UP1_gdn_conv_column.diff"
echo "apply rc=$?"

echo "--- 落地后的关键行 ---"
grep -n 's2 = ' "$F" | head -5
echo "md5_after=$(md5sum "$F" | cut -c1-12)"

echo "--- 该头的 includer（需要 touch 才能重编） ---"
grep -rln "gdn_conv.cuh" "$R/src" 2>/dev/null | sed "s|$R/||" | head -10
echo GDN_CONV_LAND_DONE
