#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
V="/mnt/c/vllm/venv-clean/Lib/site-packages/vllm"
echo "=== venv-clean 里 vllm 版本 ==="
cat /mnt/c/vllm/venv-clean/Lib/site-packages/vllm*/METADATA 2>/dev/null | grep -m2 -E '^Version|^Name'
ls -d /mnt/c/vllm/venv-clean/Lib/site-packages/vllm* 2>/dev/null | head -6
echo
echo "=== 干净 vllm 支持的 spec 方法名（grep dflash2 / dspark） ==="
grep -rl 'dflash2' "$V" 2>/dev/null | head -8
echo "--- dspark"
grep -rl 'dspark' "$V" 2>/dev/null | head -8
echo "--- 方法名注册点"
grep -rnE 'dflash2|dspark' "$V/config/speculative.py" 2>/dev/null | head -10
grep -rnE '"dflash2"|"dspark"|method ==' "$V/config/speculative.py" 2>/dev/null | head -12
echo
echo "=== data/draft_model 内容 ==="
ls -l --time-style=+%m-%d_%H:%M $J/data/draft_model 2>/dev/null | head -12
