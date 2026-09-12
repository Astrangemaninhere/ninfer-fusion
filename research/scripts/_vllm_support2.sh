#!/bin/bash
# 定点查证（不扫盘）：只看已知的单个文件与单个目录
J=/mnt/c/Users/User/Documents/ziqinzhang
SP=/mnt/c/vllm/venv-clean/Lib/site-packages/vllm/config/speculative.py
echo "=== 1) 干净 vllm 的 spec 方法名（单文件 grep） ==="
[ -f "$SP" ] && grep -nE 'dflash2|dspark|method' "$SP" | head -25 || echo "  缺 $SP"
echo
echo "=== 2) venv-clean 顶层（单目录 ls） ==="
ls -d /mnt/c/vllm/venv-clean/Lib/site-packages/vllm* 2>/dev/null | head -8
echo
echo "=== 3) data/draft_model（单目录 ls） ==="
ls -l --time-style=+%m-%d_%H:%M $J/data/draft_model 2>/dev/null | head -12
echo
echo "=== 4) 干净 vllm 里 dflash2 的实现文件是否存在于已知路径 ==="
for p in model_executor/models/dflash2.py v1/spec_decode/dflash2.py spec_decode/dflash2.py; do
  [ -f "/mnt/c/vllm/venv-clean/Lib/site-packages/vllm/$p" ] && echo "  有: $p"
done
