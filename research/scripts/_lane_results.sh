#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== ① lane 测试结果（09:22 完成）==='
cat $J/dl/lane_tests_gate.log 2>/dev/null | tail -30 | cut -c1-150
echo
echo '=== 各测试日志尾部 ==='
for f in /home/user/lt_*.log; do
  [ -f "$f" ] || continue
  printf '  %-46s ' "$(basename "$f")"
  tail -1 "$f" | cut -c1-90
done
