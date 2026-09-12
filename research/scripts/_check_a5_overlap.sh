#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== A5 的 block_rows 候选 diff（看是否与今天的 offset 修复重叠） ==="
cat "$J/_collab/A5_block_rows.diff" 2>/dev/null | head -40 | cut -c1-155
echo
echo "=== 该 diff 触及的文件/行 ==="
grep -E '^\+\+\+|^@@' "$J/_collab/A5_block_rows.diff" 2>/dev/null | head -8 | cut -c1-130
