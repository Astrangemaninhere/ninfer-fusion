#!/bin/bash
# 后台看门：等落地脚本收尾标记（最多 55 分钟），结束即回调
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/land_part_b.log
for i in $(seq 1 660); do
  if grep -qE 'LAND_PART_B_DONE|BUILD_FAIL|APPLY_FAIL|DRY_FAIL' "$L" 2>/dev/null; then break; fi
  sleep 5
done
echo "=== 收尾判读 $(date '+%H:%M:%S') ==="
grep -nE 'rc=|BUILD_FAIL|APPLY_FAIL|DRY_FAIL|LAND_PART_B_DONE' "$L" | tail -20
echo "--- 基准/一致率段 ---"
sed -n '/--- 四项目基准 ---/,$p' "$L" | tail -40
