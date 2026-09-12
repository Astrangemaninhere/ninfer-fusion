#!/bin/bash
echo "=== locate trainers:"
for root in /home/user /mnt/c/Users/User/Documents/ziqinzhang; do
  find "$root" -maxdepth 4 -name 'train_dflash2.py' -o -maxdepth 4 -name 'train_dspark.py' -o -maxdepth 4 -name 'train_dflash.py' 2>/dev/null
done | sort | head -20
echo
echo "=== grep attention mask / causal in the local trainers:"
for f in /home/user/ninfer-fusion/data/df2pilot/train_dflash2.py \
         /mnt/c/Users/User/Documents/ziqinzhang/data/df2pilot/train_dflash2.py \
         /mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py \
         /mnt/c/Users/User/Documents/ziqinzhang/dl/aeon-dflash.py ; do
  if [ -f "$f" ]; then
    echo "---- $f"
    grep -n 'causal\|attn_implementation\|is_causal\|mask\|block_pos\|position_ids' "$f" | head -40
  fi
done
