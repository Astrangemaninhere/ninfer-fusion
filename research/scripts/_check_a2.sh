#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== _collab 里 A2/A5/A3/A4 的产物 ==="
ls -lt "$J/_collab/" 2>/dev/null | head -14 | cut -c1-95
echo
echo "=== A2 报告（若已写） ==="
if [ -f "$J/_collab/A2_draft_ceiling.md" ]; then
  head -40 "$J/_collab/A2_draft_ceiling.md" | cut -c1-155
else
  echo "  （还没产出）"
fi
echo
echo "=== 编译/验证进度 ==="
tail -3 "$J/dl/rebuild_after_s35.log" | cut -c1-130
tail -3 "$J/dl/offset_family_verify.log" 2>/dev/null | cut -c1-150
