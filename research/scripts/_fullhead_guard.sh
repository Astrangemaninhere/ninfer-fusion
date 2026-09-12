#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== guard: 'requires the full proposal head' ==="
grep -rn "requires the full proposal head" $R/src $R/apps $R/include 2>/dev/null | sed "s|$R/||"
echo
echo "--- with surrounding context ---"
F=$(grep -rl "requires the full proposal head" $R/src $R/apps 2>/dev/null | head -1)
echo "file: $F"
[ -n "$F" ] && grep -n -B12 -A6 "requires the full proposal head" "$F"
echo
echo "=== OURS: every dflash2 proposal_head guard ==="
grep -rn "dflash2" $R/src/product/speculative_options.h 2>/dev/null | sed "s|$R/||" | head
echo "--- speculative_options.h 20-70 ---"
sed -n '20,70p' $R/src/product/speculative_options.h
echo
echo "=== UPSTREAM: same file for comparison ==="
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
sed -n '20,60p' $U/src/product/speculative_options.h 2>/dev/null
