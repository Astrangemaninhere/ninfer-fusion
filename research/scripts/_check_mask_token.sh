#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 引擎的 mask_token 值（训练必须与它一致） ==="
grep -rn 'mask_token' "$R/src/targets/qwen3_6/impl/" "$R/src/targets/qwen3_6_27b/impl/" 2>/dev/null | head -8 | cut -c1-150
echo
echo "=== 2) prepare_masked_block 的语义（哪些列被填 mask） ==="
grep -n -B3 -A12 'void prepare_masked_block' "$R/include/ninfer/ops/prepare_masked_block.h" 2>/dev/null | head -26 | cut -c1-140
echo
echo "=== 3) 训练脚本的 block 构造（要改的地方） ==="
grep -n -B4 -A4 'block = tok\[a:a + B\]' "$J/train_dflash2.py" | cut -c1-150
echo
echo "=== 4) 训练脚本里 BLOCK_SIZE 与 mask 相关常量 ==="
grep -nE 'BLOCK_SIZE|MASK|248070|mask' "$J/train_dflash2.py" | head -12 | cut -c1-140
