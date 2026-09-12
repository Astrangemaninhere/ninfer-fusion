#!/bin/bash
R=/home/user/ninfer-fusion
Q=$R/_orig_quarantine
echo "=== layouts_impl.h: .orig（补丁前） vs live 的差异摘要 ==="
diff -u "$Q/src/targets/qwen3_6/impl/runtime/layouts_impl.h.orig" "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" > /tmp/layouts_diff.txt 2>&1
echo "  差异行数: $(grep -c '^[+-]' /tmp/layouts_diff.txt)"
grep -n '^[-+]' /tmp/layouts_diff.txt | grep -vE '^[0-9]+:[-+]{3}' | head -30 | cut -c1-150
echo
echo "=== 关键问题：live 里新增的那些名字（is_swa_attention / parse_kv_layer_storage）在 .orig 里存在吗 ==="
for n in is_swa_attention parse_kv_layer_storage sliding_window_tokens; do
  a=$(grep -c "$n" "$Q/src/targets/qwen3_6/impl/runtime/layouts_impl.h.orig" 2>/dev/null)
  b=$(grep -c "$n" "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" 2>/dev/null)
  printf "  %-24s orig=%s live=%s\n" "$n" "$a" "$b"
done
echo
echo "=== 27b/35b 的 config.h 有没有 is_swa_attention（生产者） ==="
grep -c 'is_swa_attention' "$R/src/targets/qwen3_6_27b/impl/config.h" "$R/src/targets/qwen3_6_35b_a3b/impl/config.h" 2>/dev/null
echo
echo "=== A_s24 补丁本身是不是就加在 config.h + layouts_impl.h 上 ==="
grep -nE '^\+\+\+|^---' /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s24_window_table.diff 2>/dev/null | head -8 | cut -c1-130
