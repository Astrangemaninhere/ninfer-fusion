#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h
echo "=== dspark 的 block 行 → 草稿 token 的取法（A5 说 bf16 取 0..k-1、W8 取 1..k） ==="
sed -n '440,460p' "$F" | cat -n | sed 's/^/  440+/' | cut -c1-150
echo
echo "=== 该文件里所有 rows 相关切片（找那两条分支） ==="
grep -nE 'rows|slice\(|\.view\(|ar_hidden|proposal' "$F" | sed -n '1,40p' | cut -c1-150
echo
echo "=== W8/其他分支（grep 关键字） ==="
grep -nE 'w8|W8|rows \+ 1|rows, 1|slice\(0, 1' "$F" | head -12 | cut -c1-150
