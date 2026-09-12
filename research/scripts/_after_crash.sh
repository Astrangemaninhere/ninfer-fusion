#!/bin/bash
echo "=== WSL 状态 ==="
uptime
echo "--- /home/user 顶层 ---"
ls -1 /home/user 2>/dev/null | head -14
echo "--- 关键资产 ---"
for p in /home/user/vllm029/bin/python /home/user/models/qwen3_8_27b_nvfp4_hf/config.json /home/user/models/draft_dflash2_ref/model.safetensors /home/user/ninfer-fusion/build/apps/ninfer; do
  if [ -e "$p" ]; then echo "OK   $p"; else echo "MISS $p"; fi
done
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | head -2
echo "--- /tmp 是否被清空（已知坑） ---"
ls -1 /tmp/*.py /tmp/*.sh 2>/dev/null | head -5 || echo "(/tmp 空)"
