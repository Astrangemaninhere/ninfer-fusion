#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "########## A. 训练侧的"输出切片"（权威约定） ##########"
for f in "$J"/train_dspark.py "$J"/train_dflash2.py "$J"/train_mtp.py; do
  [ -f "$f" ] || { echo "--- $(basename $f): 不存在 ---"; continue; }
  echo "--- $(basename $f) ---"
  grep -nE 'out\[:, *1:\]|\[:, *1:\]|lm_head\(|block_pos|arange\(|teacher_idx|shift|target_shift' "$f" | head -14 | cut -c1-150
done
echo
echo "########## B. 推理侧：三个后端的"取列/跳过 anchor"点 ##########"
for f in dflash_impl.h dflash2_impl.h mtp_impl.h; do
  P=$R/src/targets/qwen3_6/impl/runtime/$f
  [ -f "$P" ] || { echo "--- $f: 不存在 ---"; continue; }
  echo "--- $f ---"
  grep -nE 'source_column_offset|source_pitch|hidden \* element_bytes|columns? 1|slice\(1|slice\(0, 1|\[:, *1:\]|skip.*anchor|anchor column|packed' "$P" | head -14 | cut -c1-155
done
echo
echo "########## C. 各后端草稿的宽度/anchor 语义声明 ##########"
grep -rnE 'anchor column|bonus token|column 0|columns 1|masked column' "$R/src/targets/qwen3_6/impl/runtime/"*.h 2>/dev/null | head -14 | cut -c1-160
