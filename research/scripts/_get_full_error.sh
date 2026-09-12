#!/bin/bash
# 从启动器日志里捞出完整 traceback（含文件名与原因）
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/vllm_launch.log
echo '=== 含 SafetensorError / Traceback / FileNotFound / No such 的行 ==='
grep -nE 'SafetensorError|Traceback|FileNotFound|No such file|incomplete metadata|Repository|IndexError' "$L" 2>/dev/null | tail -20 | cut -c1-220
echo
echo '=== 错误附近 40 行（去重后）==='
awk '/SafetensorError|Traceback/{found=NR} END{}' "$L" >/dev/null
N=$(grep -nE 'SafetensorError' "$L" | head -1 | cut -d: -f1)
if [ -n "$N" ]; then
  awk -v s=$((N-40)) -v e=$((N+6)) 'NR>=s && NR<=e {printf "%5d| %s\n", NR, $0}' "$L" | cut -c1-200 | uniq
else
  echo "（日志里没找到 SafetensorError，可能只出现在 err）"
fi
echo
echo '=== err 文件 ==='
grep -nE 'SafetensorError|Traceback|not fully|incomplete' /mnt/c/Users/User/Documents/ziqinzhang/dl/vllm_launch.err 2>/dev/null | tail -10 | cut -c1-200
