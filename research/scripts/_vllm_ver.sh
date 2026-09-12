#!/bin/bash
# 定点核对：单文件 grep + 单目录 ls（不扫盘）
SP=/mnt/c/vllm/venv-clean/Lib/site-packages/vllm/config/speculative.py
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) venv-clean 的 speculative.py 里方法名全集（单文件） ==="
grep -nE 'SpeculativeMethod|Literal\[|dflash2|dflash|dspark' "$SP" 2>/dev/null | head -25
echo
echo "=== 2) venv-clean 的 site-packages 顶层（单目录，只看相关） ==="
ls -1 /mnt/c/vllm/venv-clean/Lib/site-packages 2>/dev/null | grep -iE 'vllm|1cat|dflash|dspark' | head -10
echo
echo "=== 3) vllm 里与 dflash/dspark 有关的模型或 spec 模块（单目录 ls） ==="
ls -1 /mnt/c/vllm/venv-clean/Lib/site-packages/vllm/model_executor/models 2>/dev/null | grep -iE 'dflash|dspark|mtp' | head
ls -1 /mnt/c/vllm/venv-clean/Lib/site-packages/vllm/v1/spec_decode 2>/dev/null | head -12
echo
echo "=== 4) 老 venv 里是否有 1Cat fork ==="
ls -1 /mnt/c/vllm/venv/Lib/site-packages 2>/dev/null | grep -iE '1cat|vllm' | head -6
echo
echo "=== 5) 工作区 1Cat-vLLM 是否为 git 仓库（单目录） ==="
ls -1d $J/1Cat-vLLM/.git 2>/dev/null && git -C $J/1Cat-vLLM log --oneline -3 2>/dev/null
echo "--- 该 fork 的 dflash2 支持（单文件）"
grep -nE 'dflash2|dflash|dspark' $J/1Cat-vLLM/vllm/config/speculative.py 2>/dev/null | head -15
