#!/bin/bash
cd /mnt/c/Users/User/Documents/ziqinzhang
echo "=== 含 N1..N7 / 八项 的文件 ==="
grep -rlE '八项|N1[^0-9a-zA-Z]|N7' _collab/*.md *.md 2>/dev/null | head -12
echo "=== 定义处上下文 ==="
for f in $(grep -rlE '八项' _collab/*.md *.md 2>/dev/null | head -4); do
  echo "----- $f"
  grep -nE '八项|^#{2,4} *N[0-9]|N[1-7][^0-9a-zA-Z]' "$f" | head -25
done
