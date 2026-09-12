#!/bin/bash
# 找 github 仓库 URL 线索（定点，不扫盘）
{
  echo "=== 树内/工作区出现的 github URL ==="
  grep -rhoE 'https?://(www\.)?github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+' \
    /home/user/ninfer-fusion/README.md /home/user/ninfer-fusion/docs \
    /home/user/ninfer-fusion/.gitmodules \
    /mnt/c/Users/User/Documents/ziqinzhang/*.md /mnt/c/Users/User/Documents/ziqinzhang/_collab/*.md 2>/dev/null \
    | sort | uniq -c | sort -rn | head -20
  echo
  echo "=== 含 ninfer-fusion 字样的 git/远端线索 ==="
  grep -rniE 'ninfer-fusion[^ ]{0,40}(\.git|github|origin|remote)' \
    /home/user/ninfer-fusion/README.md /home/user/ninfer-fusion/docs \
    /mnt/c/Users/User/Documents/ziqinzhang/_TODO.md 2>/dev/null | head -12
  echo
  echo "=== ninfer-fusion-repo 里有什么（它是什么）==="
  ls /home/user/ninfer-fusion-repo 2>/dev/null | head -8
  ls -la /home/user/ninfer-fusion-repo 2>/dev/null | head -6
  echo
  echo "=== 是否有 git bundle / 归档可恢复历史 ==="
  ls -l /home/user/*.bundle /home/user/ninfer-fusion-repo/*.bundle 2>/dev/null | head -5
  echo
  echo "=== 体积守卫结果（上一次 git_prep）==="
  cat /mnt/c/Users/User/Documents/ziqinzhang/dl/_git_sync.txt 2>/dev/null
} > /mnt/c/Users/User/Documents/ziqinzhang/dl/_urlhunt.txt 2>&1
echo written
