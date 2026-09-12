#!/bin/bash
# 检查 0.29 的 dflash2 是否默认带 lookup/LABD（要有开关则关掉，保证与 ninfer 同口径）
V=/home/user/vllm029/lib/python3.12/site-packages/vllm
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) dflash2 目录内容 ==="
ls -1 $V/v1/worker/gpu/spec_decode/dflash2/ 2>/dev/null
echo
echo "=== 2) lookup 是否存在及其开关（单文件 grep） ==="
grep -rnE 'lookup|LABD|look_ahead|VLLM_DFLASH2|env|getenv' $V/v1/worker/gpu/spec_decode/dflash2/*.py 2>/dev/null | head -20
echo
echo "=== 3) 方法名注册（单文件） ==="
grep -nE 'dflash2|"dflash"|DraftModelTypes|Literal' $V/config/speculative.py 2>/dev/null | head -12
echo
echo "=== 4) 安装进度 ==="
tail -4 $J/dl/vllm029_install2.log 2>/dev/null
