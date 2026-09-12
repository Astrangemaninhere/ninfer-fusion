#!/bin/bash
# 诊断：谁有 CRLF？为什么 S52 的 diff 报 "different line endings"？
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo '=== 树里相关文件的换行符 ==='
for f in src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
         src/targets/qwen3_6/impl/runtime/dflash_impl.h \
         tests/targets/qwen3_6/test_context_store.cpp; do
  p="$R/$f"
  crlf=$(grep -c $'\r' "$p" 2>/dev/null || echo -1)
  total=$(wc -l < "$p" 2>/dev/null || echo -1)
  echo "  CRLF行=$crlf / 总行=$total  ->  $f"
done
echo
echo '=== S52 的 diff 自身换行符 ==='
d="$J/E9_s52_dflash2_k_slice.diff"
crlf=$(grep -c $'\r' "$d"); echo "  原始 CRLF行=$crlf"
tr -d '\r' < "$d" > /tmp/s52_lf.diff
crlf2=$(grep -c $'\r' /tmp/s52_lf.diff || true); echo "  strip后 CRLF行=$crlf2"
echo
echo '=== 原始 S52 diff 里 dflash2_impl.h 的 hunk 头与首行 ==='
grep -n 'dflash2_impl.h' "$d" | head -5
awk 'NR>=1 && /dflash2_impl.h/{found=1} found && c<8 {print; c++}' "$d" | cat -A | head -10 | cut -c1-120
echo
echo '=== _collab 里 S52/E9 相关文件 ==='
ls -1 "$J" | grep -iE 's52|e9' 
