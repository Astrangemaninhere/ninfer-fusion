#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/vref_df2.err
echo "=== 候选根因行 ==="
grep -nE 'out of memory|OOM|CUDA error|unsupported|not supported|ValueError|AssertionError|TypeError|KeyError|raise [A-Z]|Error:' "$L" 2>/dev/null | head -20
echo
echo "=== 首个 Traceback 之前 30 行 ==="
N=$(grep -n 'Traceback (most recent call last)' "$L" 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "${N:-}" ]; then
  S=$(( N > 32 ? N - 32 : 1 ))
  sed -n "${S},${N}p" "$L" | head -36
else
  echo "(没有 Traceback)"
fi
echo
echo "=== 日志中与 dflash/draft 相关的行 ==="
grep -niE 'dflash|draft|speculat' "$L" 2>/dev/null | tail -12
