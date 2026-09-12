#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
for i in $(seq 1 12); do
  sleep 20
  if grep -q 'VERIFY_EQUIV_DONE' "$J/dl/verify_equivalence.log" 2>/dev/null; then
    echo "=== 实验完成 ==="; break
  fi
  echo "  [${i}] $(tail -1 /home/user/ve_*.log 2>/dev/null | tail -1 | cut -c1-90)"
done
echo
echo "=== 原始输出 ==="
cat "$J/dl/verify_equivalence.log" 2>/dev/null | tail -20 | cut -c1-180
echo
echo "=== 判定文件 ==="
cat "$J/_collab/M_verify_equivalence.md" 2>/dev/null | head -20
