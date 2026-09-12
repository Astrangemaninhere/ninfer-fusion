#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== 1) dspark/dflash 的训练脚本在哪 ==="
ls -t "$J"/train_*.py "$J"/dspark*.py 2>/dev/null | head -8 | sed 's/^/  /'
ls -t "$J"/_scratch/*/train_*.py 2>/dev/null | head -5 | sed 's/^/  (scratch) /'
find "$R/tools" -iname '*dspark*' -o -iname '*dflash*' 2>/dev/null | head -8 | sed 's/^/  repo: /'
echo
echo "=== 2) 训练脚本里的 rope 实现（有没有 yarn / 只纯 rope？） ==="
for f in $(ls -t "$J"/train_dspark*.py "$J"/train_dflash*.py 2>/dev/null | head -3); do
  echo "--- $(basename $f) ---"
  grep -nE 'rope|yarn|inv_freq|theta|rotary' "$f" | head -12 | cut -c1-140
done
echo
echo "=== 3) 该脚本如何构造位置/block（含 anchor 列？） ==="
for f in $(ls -t "$J"/train_dspark*.py 2>/dev/null | head -1); do
  echo "--- $(basename $f) ---"
  grep -nE 'block|anchor|width|shift|ids|target' "$f" | head -20 | cut -c1-140
done
