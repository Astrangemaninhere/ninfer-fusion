#!/bin/bash
B=/home/user/ninfer-fusion
M=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "=== 在两棵树里找位置直方图的定义与累加点 ==="
for T in "$B" "$M"; do
  echo "--- $T ---"
  grep -rn 'accepted_per_position\|accepted by pos' "$T/src" "$T/apps" 2>/dev/null | head -6 | cut -c1-150
done
echo
echo "=== accept_pos / per-position 的累加（广搜） ==="
grep -rn 'per_position\|accept_pos\|accepted_by_pos' "$B/src" 2>/dev/null | head -8 | cut -c1-150
