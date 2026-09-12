#!/bin/bash
# 检查 KV dump 是否已产出；齐则直接对照，不齐则补跑
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
A=$J/dl/kvA
B=$J/dl/kvB
na=$(ls "$A" 2>/dev/null | wc -l)
nb=$(ls "$B" 2>/dev/null | wc -l)
echo "dump 文件数: plain=$na  spec=$nb"
if [ "$na" -gt 0 ] && [ "$nb" -gt 0 ]; then
  echo "--- 两臂都有 dump，直接对照 ---"
  python3 "$J/_kv_cmp.py" "$A" "$B" 2>&1 | tail -16
else
  echo "--- 有缺失，补跑 dump+对照 ---"
  bash "$J/_kv_dump_cmp_run.sh" 2>&1 | tail -20
fi
echo "--- 运行日志尾部 ---"
tail -2 "$J/dl/kvdump_plain.log" 2>/dev/null
tail -2 "$J/dl/kvdump_spec.log" 2>/dev/null
