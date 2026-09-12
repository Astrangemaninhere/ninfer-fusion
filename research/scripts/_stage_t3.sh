#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
mkdir -p "$D"
for f in \
  src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
  src/targets/qwen3_6/impl/runtime/dflash_impl.h \
  src/targets/qwen3_6/impl/runtime/dflash_context.h \
  src/targets/qwen3_6/impl/runtime/dflash_context_impl.h
do
  b=$(basename "$f")
  tr -d '\r' < "$R/$f" > "$D/$b"
  echo "$b $(wc -l < "$D/$b")"
done
echo "=== find program_impl / speculative ==="
find "$R/src" -name 'program_impl.h' -o -name 'speculative*' 2>/dev/null | head -30
echo "=== find train scripts with dflash/dspark ==="
grep -rl --include=*.py -iE 'dflash|dspark' "$R/tools" 2>/dev/null | head -40
echo "=== git status of tree ==="
cd "$R" && git rev-parse --short HEAD 2>&1 | head -2
