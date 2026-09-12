#!/bin/bash
# 三后端 verify ingress 的位置/列数约定对照：S48/E6 修了 dspark 的 pos k→k+1，
# 兄弟后端是否一致？（同一 bug 家族的最后嫌疑点）
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/program_impl.h
for L in 11965 12161 12393; do
  echo "=== program_impl.h 从 $L 起 24 行 ==="
  awk -v s=$L 'NR>=s && NR<s+24 { printf "%5d| %s\n", NR, $0 }' "$F" \
    | grep -E 'position|valid_columns|extent|slot|draft' | cut -c1-135
  echo
done
echo '=== 三行 target_valid_columns 赋值原文 ==='
grep -n 'target_valid_columns\[row\]' "$F" | cut -c1-140
