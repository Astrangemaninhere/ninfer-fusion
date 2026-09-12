#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== train_dflash2.py 里 mask-block / target-shift 的默认与用法 ==="
grep -nE 'add_argument|mask.block|mask_block|target.shift|target_shift' $J/train_dflash2.py | head -30
echo
echo "=== 实际构造 block 的代码 ==="
grep -nE '248070|block\[|block =|mask' $J/train_dflash2.py | head -20
