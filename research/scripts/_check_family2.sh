#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "########## 1. dflash2 推理侧：packed 是否跳过第 0 列 ##########"
sed -n '300,320p' "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" | cat -n | sed 's/^/  300+/' | cut -c1-150
echo
echo "########## 2. dflash2 训练侧：block 位置 ↔ 目标行的配对（关键！） ##########"
sed -n '425,445p' "$J/train_dflash2.py" | cat -n | sed 's/^/  425+/' | cut -c1-160
echo
echo "########## 3. ids16 的语义（load_cache_seq） ##########"
grep -n -A12 'def load_cache_seq' "$J/train_dflash2.py" | head -24 | cut -c1-150
echo
echo "########## 4. mtp_impl.h 的草稿取法（机制不同，单独看） ##########"
grep -nE 'packed|proposal|draft|hidden|slice\(1|arange|position' "$R/src/targets/qwen3_6/impl/runtime/mtp_impl.h" | head -18 | cut -c1-150
