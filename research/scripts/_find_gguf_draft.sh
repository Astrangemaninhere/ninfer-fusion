#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) data/ 顶层（单目录） ==="
ls -1 --time-style=+%m-%d_%H:%M -l $J/data 2>/dev/null | head -30
echo
echo "=== 2) GGUF 文件（已知目录定点） ==="
ls -l --time-style=+%m-%d_%H:%M $J/data/*.gguf $J/models/*.gguf $J/dl/*.gguf /home/user/*.gguf /home/user/models/*.gguf 2>/dev/null | head -12
echo
echo "=== 3) HF 缓存里的 dflash2 草稿（单目录过滤） ==="
for c in /home/user/.cache/huggingface/hub /mnt/c/Users/User/.cache/huggingface/hub; do
  if [ -d "$c" ]; then
    echo "--- $c"
    ls -1 "$c" 2>/dev/null | grep -iE 'dflash|dspark|qwen3.8|qwen3_8|27b|incoai' | head -12
  fi
done
echo
echo "=== 4) 报告与备忘里的 incoai/GGUF 线索（单文件） ==="
grep -rniE 'incoai|gguf|下载了|已下载|download' $J/_collab/build/R1_dflash2_acceptance_findings.md $J/_collab/build/R2_vllm_same_collapse.md $J/_collab/build/R3_1cat_vllm_verdict.md $J/_labd_1cat_research.md 2>/dev/null | head -20
