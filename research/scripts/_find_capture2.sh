#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '=== 我们：capture_layer / capture_positions 的全部出现 ==='
grep -rn 'capture_layer\|capture_positions' "$R/src" 2>/dev/null | grep -v '\.orig' | cut -c1-160
echo
echo '=== 上游：capture_layer / capture_positions 的全部出现 ==='
grep -rn 'capture_layer\|capture_positions' "$U/src" 2>/dev/null | cut -c1-160
