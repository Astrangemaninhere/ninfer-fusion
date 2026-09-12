#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/vllm_fg2.log
echo '=== 根因（EngineCore 的异常栈，取 EngineDeadError 之前 60 行里的非重复行）==='
N=$(grep -n 'EngineDeadError' "$L" | head -1 | cut -d: -f1)
if [ -n "$N" ]; then
  awk -v s=$((N-70)) -v e=$N 'NR>=s && NR<=e {print}' "$L" | grep -vE '^\s*$' | uniq | tail -45 | cut -c1-190
else
  echo '（未找到 EngineDeadError）'
fi
echo
echo '=== 关键词扫描 ==='
grep -nE 'Error|error|Exception|assert|CUDA|dtype|quant|not supported|Unsupported' "$L" | grep -viE 'ERROR\] `|NativeCommandError' | head -20 | cut -c1-190
