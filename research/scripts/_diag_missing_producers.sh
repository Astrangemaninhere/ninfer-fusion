#!/bin/bash
R=/home/user/ninfer-fusion
M=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
for name in is_swa_attention parse_kv_layer_storage sliding_window_attention; do
  echo "=== $name ==="
  echo "  build tree:"
  grep -rn "$name" "$R/src" "$R/include" 2>/dev/null | grep -vE 'layouts_impl.h' | head -4 | cut -c1-140
  echo "  mirror:"
  grep -rn "$name" "$M/src" "$M/include" 2>/dev/null | grep -vE 'layouts_impl.h' | head -4 | cut -c1-140
  echo "  协作区:"
  grep -rln "$name" "$J/_collab" 2>/dev/null | head -3
done
echo
echo "=== layouts_impl.h 出错处上下文 ==="
sed -n '165,176p' "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" | cat -n | sed 's/^/  165+/' | cut -c1-140
echo "  --- 1035-1042 ---"
sed -n '1035,1042p' "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" | cat -n | sed 's/^/  1035+/' | cut -c1-140
echo
echo "=== 隔离区里的 layouts_impl.h.orig 是不是同一个文件的前一版 ==="
ls -l "$R/_orig_quarantine/src/targets/qwen3_6/impl/runtime/layouts_impl.h.orig" 2>/dev/null | awk '{print "  ", $5, $NF}'
