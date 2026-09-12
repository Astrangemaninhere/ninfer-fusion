#!/bin/bash
F=/mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py
echo "=== head 1-58:"
sed -n '1,58p' "$F"
echo
echo "=== grep ctx / window / CTX:"
grep -n 'ctx\|CTX\|window\|WIN' "$F" | head -50
