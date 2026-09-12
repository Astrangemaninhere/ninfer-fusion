#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
mkdir -p "$D"
for f in \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  src/targets/qwen3_6/impl/runtime/speculative_target_impl.h \
  src/product/speculative_options.h
do
  b=$(basename "$f")
  tr -d '\r' < "$R/$f" > "$D/$b"
  echo "$b $(wc -l < "$D/$b")"
done
echo "=== tools tree (depth2) ==="
find "$R/tools" -maxdepth 2 -type d | sort
echo "=== any py mentioning target-shift / ids16 ==="
grep -rln --include=*.py -e 'target-shift' -e 'target_shift' -e 'ids16' "$R" 2>/dev/null | grep -v '/build/' | head -30
echo "=== files with 'dflash2' in name anywhere ==="
find "$R" /mnt/c/Users/User/Documents/ziqinzhang -iname '*dflash2*' -not -path '*/build/*' 2>/dev/null | head -40
