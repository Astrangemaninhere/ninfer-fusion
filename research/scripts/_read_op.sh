#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== offset_probe 日志 ==="
tail -20 "$J/dl/offset_probe.log" 2>/dev/null | cut -c1-200
echo
echo "=== 判定文件 ==="
cat "$J/_collab/M_offset_probe.md" 2>/dev/null | tail -14
echo
echo "=== 每档的 prompt tokens 与 ids（原始） ==="
for f in /home/user/op_s_plain.log /home/user/op_s_mtp.log /home/user/op_m_plain.log /home/user/op_m_mtp.log /home/user/op_l_plain.log /home/user/op_l_mtp.log; do
  [ -f "$f" ] || { echo "  $(basename $f): 缺失"; continue; }
  pt=$(grep -oE 'prompt tokens[[:space:]]+[0-9]+' "$f" | tail -1)
  ids=$(grep -E '^tokens[[:space:]]+generated ids' "$f" | tail -1 | awk '{print $4, $5, $6, $7}')
  echo "  $(basename $f): $pt ids0-3=[$ids]"
done
