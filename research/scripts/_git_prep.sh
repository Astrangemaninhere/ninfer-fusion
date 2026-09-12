#!/bin/bash
# 为"同步到 GitHub"做准备：在 70MB 源码树上初始化仓库并提交（带体积守卫，绝不纳入大件）
set -u
R=/home/user/ninfer-fusion
OUT=/mnt/c/Users/User/Documents/ziqinzhang/dl/_git_sync.txt
{
  echo "=== GitHub 同步准备 起 $(date '+%H:%M:%S') ==="
  cd "$R" || exit 3

  if [ -d .git ]; then
    echo "已是 git 仓库，跳过 init"
  else
    git init -q -b main 2>&1 | head -3
    echo "已 git init -b main"
  fi
  git config user.name  "$(git config --global user.name 2>/dev/null || echo 'ninfer-dev')"
  git config user.email "$(git config --global user.email 2>/dev/null || echo 'ninfer-dev@local')"
  echo "  本地身份: $(git config user.name) <$(git config user.email)>"

  echo "=== 体积守卫：先看会被纳入的文件数与总字节 ==="
  git add -A --dry-run 2>/dev/null | wc -l | sed 's/^/  待纳入条目数 = /'
  # 列出最大的 20 个未忽略文件（防止误纳大件）
  echo "  未被 .gitignore 排除的最大文件 Top 20:"
  git ls-files -o --exclude-standard 2>/dev/null | head -200000 | while read -r f; do
    [ -f "$f" ] && stat -c '%s %n' "$f"
  done | sort -rn | head -20 | awk '{printf "    %10.2f MB  %s\n", $1/1048576, $2}'
  # 守卫：任何未被排除的单文件 > 200MB 就中止
  big=$(git ls-files -o --exclude-standard 2>/dev/null | head -200000 | while read -r f; do
    [ -f "$f" ] && stat -c '%s' "$f"
  done | sort -rn | head -1)
  big=${big:-0}
  if [ "$big" -gt 209715200 ]; then
    echo "ABORT: 存在未忽略的大文件 ($((big/1048576)) MB) ⇒ 先补 .gitignore，不提交"
    exit 9
  fi
  echo "  守卫通过（最大未忽略文件 $((big/1048576)) MB）"
} > "$OUT" 2>&1
echo written
