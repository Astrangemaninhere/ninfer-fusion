#!/bin/bash
# 探明：git 远端/状态、仓库大小与可提交物；同时把 dwm/GPU 查询交给 PowerShell（另发）
set -u
R=/home/user/ninfer-fusion
{
  echo "=== git 远端 ==="
  git -C "$R" remote -v 2>&1 | head -5
  echo "=== 当前分支与最近提交 ==="
  git -C "$R" branch --show-current 2>&1
  git -C "$R" log --oneline -3 2>&1
  echo "=== 工作区状态（摘要）==="
  git -C "$R" status --porcelain 2>&1 | head -30
  echo "  改动文件数 = $(git -C "$R" status --porcelain 2>/dev/null | wc -l)"
  echo "=== 仓库体积 ==="
  du -sh "$R" 2>/dev/null
  du -sh "$R/.git" 2>/dev/null
  echo "=== 是否存在 .gitignore 及是否忽略大件 ==="
  [ -f "$R/.gitignore" ] && head -20 "$R/.gitignore" || echo "(无 .gitignore)"
  echo "=== 构建产物是否被跟踪（应为空）==="
  git -C "$R" ls-files build 2>/dev/null | head -3
  echo "=== 我加的文档/脚本（候选提交物）==="
  ls -l "$R/docs/maintainer/speculative-dflash2-status.md" 2>/dev/null
  echo "=== C 侧数据体积（这才是""大量数据""）==="
  du -sh /mnt/c/Users/User/Documents/ziqinzhang/dl 2>/dev/null
  du -sh /mnt/c/Users/User/Documents/ziqinzhang/_collab 2>/dev/null
  ls /mnt/c/Users/User/Documents/ziqinzhang/dl | wc -l
} > /mnt/c/Users/User/Documents/ziqinzhang/dl/_gitprobe.txt 2>&1
echo written
