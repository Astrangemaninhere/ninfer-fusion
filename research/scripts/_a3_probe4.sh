#!/bin/bash
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "### grep headwise in _collab (files only)"
grep -rl 'headwise' $C 2>/dev/null | head -30
echo "### git repo on windows tree?"
ls -d /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/.git 2>/dev/null && (cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo && git status --short | head -20)
echo "### ninfer-fusion-repo sigmoid_mul.cpp: has headwise?"
grep -n 'headwise' /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/src/ops/wrapper/sigmoid_mul.cpp 2>/dev/null || echo "no headwise in repo copy"
echo "### ninfer-fusion-repo wrapper lines"
wc -l /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/src/ops/wrapper/sigmoid_mul.cpp 2>/dev/null
echo "### A3_mkpatch.py"
cat -n $C/A3_mkpatch.py
echo "### A3_spark_newops.md (head 100)"
head -100 $C/A3_spark_newops.md
