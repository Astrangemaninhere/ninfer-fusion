#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/land_part_b.log
for i in $(seq 1 115); do
  if grep -qE 'LAND_PART_B_DONE|BUILD_FAIL|APPLY_FAIL|DRY_FAIL' "$L" 2>/dev/null; then break; fi
  sleep 5
done
echo "=== 收尾判读 ==="
grep -nE 'rc=|BUILD_FAIL|APPLY_FAIL|DRY_FAIL|LAND_PART_B_DONE|APPLY|^ *[0-9]+ \|' "$L" | tail -40
echo "--- 末 25 行 ---"
tail -25 "$L"
