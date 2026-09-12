#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_todo_131.py"
echo "--- shift0 重训脚本已写（不运行） ---"
ls -l "$J/_train_df2_shift0.bat" | awk '{print "  ", $5, $NF}'
grep -c 'target-shift 0' "$J/_train_df2_shift0.bat" | sed 's/^/  target-shift 0 出现次数: /'
echo
echo "=== 等编译 + 验证（最多 25 分钟，每 90 秒报一次） ==="
for i in $(seq 1 16); do
  sleep 90
  if grep -q 'OFFSET_FAMILY_VERIFY_DONE' "$J/dl/offset_family_verify.log" 2>/dev/null; then
    echo "  验证完成"; break
  fi
  nb=$(pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1 && echo busy || echo idle)
  bin=$(ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | awk '{print $6}')
  echo "  [$i] nvcc=$nb  ninfer=$bin"
done
echo
echo "=== 验证链输出 ==="
tail -28 "$J/dl/offset_family_verify.log" 2>/dev/null | cut -c1-180
