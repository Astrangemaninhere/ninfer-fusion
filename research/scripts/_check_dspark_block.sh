#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/train_dspark.py
echo "=== block_ids / block_pos 的构造（判定第 0 列是不是 anchor） ==="
grep -nE 'block_ids|block_pos|block_hidden|B =|block_size|anchor' "$F" | head -30 | cut -c1-150
echo
echo "=== 训练循环：目标 token 与 block 列的配对 ==="
grep -nE 'targets|labels|shift|\[:, *1:|a \+|a:' "$F" | head -24 | cut -c1-150
echo
echo "=== config 里声明的 block 宽度 ==="
grep -nE "'block|block_size|n_block|B\b" "$F" | head -14 | cut -c1-140
