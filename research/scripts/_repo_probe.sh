#!/bin/bash
# 找已知的 git 仓库与远端；量各候选数据的体积（决定推什么、用不用 LFS）
{
  echo "=== 候选路径是否有 .git ==="
  for p in /home/user/ninfer-fusion /home/user/ninfer-fusion-repo /mnt/c/Users/User/Documents/ziqinzhang; do
    if [ -d "$p/.git" ]; then
      echo "  $p : 是 git 仓库"
      echo "     remote: $(git -C "$p" remote -v 2>&1 | head -2 | tr '\n' ' ')"
      echo "     branch: $(git -C "$p" branch --show-current 2>&1)  最近: $(git -C "$p" log --oneline -2 2>&1 | tr '\n' ' | ')"
      echo "     未提交改动数: $(git -C "$p" status --porcelain 2>/dev/null | wc -l)"
    else
      echo "  $p : 非 git 仓库"
    fi
  done
  echo
  echo "=== 体积 ==="
  echo "  ninfer-fusion 总: $(du -sh /home/user/ninfer-fusion 2>/dev/null | cut -f1)"
  echo "    build/:        $(du -sh /home/user/ninfer-fusion/build 2>/dev/null | cut -f1)"
  echo "    源码(src+include+apps+tools+docs): $(du -sh --exclude=build /home/user/ninfer-fusion 2>/dev/null | cut -f1)"
  for d in src include apps tools docs jinfer tools/freq_corpus tools/calib tests; do
    [ -d "/home/user/ninfer-fusion/$d" ] && echo "    $d: $(du -sh /home/user/ninfer-fusion/$d 2>/dev/null | cut -f1)"
  done
  echo "  C 侧工作区 ziqinzhang: $(du -sh /mnt/c/Users/User/Documents/ziqinzhang 2>/dev/null | cut -f1)"
  echo "    dl/:      $(du -sh /mnt/c/Users/User/Documents/ziqinzhang/dl 2>/dev/null | cut -f1)"
  echo "    _collab/: $(du -sh /mnt/c/Users/User/Documents/ziqinzhang/_collab 2>/dev/null | cut -f1)"
  echo "    dl 顶层文件数: $(ls /mnt/c/Users/User/Documents/ziqinzhang/dl 2>/dev/null | wc -l)"
  echo "  models:  $(du -sh /home/user/models 2>/dev/null | cut -f1)"
  echo
  echo "=== 是否有 git/gh 凭据 ==="
  command -v git >/dev/null && echo "  git: $(git --version)"
  command -v gh >/dev/null && echo "  gh: $(gh --version 2>&1 | head -1)" || echo "  gh: 未安装"
  ls -la ~/.ssh 2>/dev/null | head -5
  git config --global --get user.name 2>/dev/null || echo "  git user.name 未设"
  git config --global --get user.email 2>/dev/null || echo "  git user.email 未设"
  [ -f ~/.git-credentials ] && echo "  ~/.git-credentials 存在（HTTPS 凭据）" || echo "  ~/.git-credentials 不存在"
} > /mnt/c/Users/User/Documents/ziqinzhang/dl/_repo_probe.txt 2>&1
echo written
