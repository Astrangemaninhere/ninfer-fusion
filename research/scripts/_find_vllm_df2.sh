#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 哪些日志里出现过 method=dflash2 ==="
grep -rlE '"method" *: *"dflash2"' "$D/dl/" "$D"/*.py "$D"/*.bat "$D"/*.txt 2>/dev/null | head -20
echo
echo "=== 2) 所有日志里 method 的取值统计 ==="
grep -rhoE '"method" *: *"[a-z0-9]+"' "$D/dl/"*.log 2>/dev/null | sort | uniq -c | sort -rn | head
echo
echo "=== 3) dl/ 下按时间最近的 12 个日志（找干净环境那次） ==="
ls -lt "$D/dl/"*.log 2>/dev/null | head -12
echo
echo "=== 4) 干净环境相关脚本 vllm_clean ==="
ls -l "$D"/_vllm_clean*.py "$D"/_run_vllm_clean*.bat 2>/dev/null
grep -nE 'dflash2|method|spec' "$D"/_vllm_clean.py 2>/dev/null | head -12
