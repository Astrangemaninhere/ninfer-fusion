#!/bin/bash
# 给 vLLM 启动器加 --max-num-seqs 32（否则 Mamba 缓存块不够）
F=/mnt/c/Users/User/Documents/ziqinzhang/_vllm_ref3.py
cp "$F" "$F.bak"
sed -i 's|"--max-model-len", "4096",|"--max-model-len", "4096",\n        "--max-num-seqs", "32",|' "$F"
echo '--- 修改后的 args 段 ---'
grep -n -A 10 'args = \[' "$F" | head -14
rm -f "$F.bak"
