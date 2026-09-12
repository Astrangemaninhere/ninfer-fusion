#!/bin/bash
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=$J/dl/streamer_install2.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 补齐 runai 依赖 $(date '+%H:%M:%S') ==="
for m in humanize; do
  echo "--- 装 $m"
  $V -m pip install -i $IDX --no-deps "$m" 2>&1 | tail -3
done
echo "--- 导入验证（最多迭代 4 次补缺）---"
for i in 1 2 3 4; do
  out=$($V -c "import runai_model_streamer; print('OK')" 2>&1 | tail -2)
  echo "$out"
  echo "$out" | grep -q OK && break
  mod=$(echo "$out" | grep -oE "No module named '[^']+'" | sed "s/No module named //;s/'//g" | head -1)
  [ -z "$mod" ] && break
  echo "  -> 补装 $mod"
  $V -m pip install -i $IDX --no-deps "$mod" 2>&1 | tail -2
done
echo "--- 最终 ---"
$V -c "import runai_model_streamer; print('runai ok')" 2>&1 | tail -2
$V -m pip list 2>/dev/null | grep -iE 'runai|humanize'
echo STREAMER_OK_DONE
