#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## 1) supervise_build.ps1 ##########"
head -50 $J/supervise_build.ps1 2>/dev/null
echo
echo "########## 2) build_sm70.bat ##########"
head -35 $J/build_sm70.bat 2>/dev/null
echo
echo "########## 3) _z2_dl_wheelhouse.ps1 ##########"
head -30 $J/_z2_dl_wheelhouse.ps1 2>/dev/null
echo
echo "########## 4) 工作区根里含 'vllm' 且像构建脚本的（单目录 glob） ##########"
for f in $J/*.bat $J/*.ps1 $J/*.sh; do
  if grep -lqiE 'bdist_wheel|setup\.py|vllm.*build|build.*vllm' "$f" 2>/dev/null; then echo "  $f"; fi
done
