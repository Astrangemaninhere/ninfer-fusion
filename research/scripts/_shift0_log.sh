#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 09-10 22:30 之后改过、且与训练有关的日志 ==="
for f in $J/dl/*.log; do
  t=$(stat -c %Y "$f" 2>/dev/null)
  if [ "$t" -gt 1757520000 ] 2>/dev/null; then :; fi
done
ls -lt --time-style=+%m-%d_%H:%M $J/dl/ 2>/dev/null | awk '$6 ~ /09-1[01]/' | grep -iE 'train|shift|s0|df2' | head -12
echo
echo "=== 直接找训练日志候选 ==="
ls -lt --time-style=+%m-%d_%H:%M $J/dl/train*.log $J/train*.log $J/dl/*shift*.log 2>/dev/null | head -8
echo
echo "=== 若有日志，抓 banner/命令行/step 行 ==="
for L in $J/dl/train-dflash2.log $J/dl/train_shift0.log $J/dl/shift0.log; do
  if [ -f "$L" ]; then
    echo "--- $L ($(stat -c %y "$L" | cut -c1-16))"
    grep -nE 'mask|shift|ctx|steps|step [0-9]|loss' "$L" 2>/dev/null | head -8
    echo "   末行: $(tail -1 "$L")"
  fi
done
