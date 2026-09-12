#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) data/draft_dflash2_ref 内容 ==="
ls -l --time-style=+%m-%d_%H:%M $J/data/draft_dflash2_ref 2>/dev/null | head -15
echo "--- config.json ---"
cat $J/data/draft_dflash2_ref/config.json 2>/dev/null | head -40
echo
echo "=== 2) 同目录下有无下载/转换日志 ==="
ls -lt --time-style=+%m-%d_%H:%M $J/data/draft_dflash2_ref/*.log $J/data/draft_dflash2_ref/*.txt $J/dl/*dflash2_ref* 2>/dev/null | head -8
echo
echo "=== 3) HF 缓存里是否也有 incoai 的 dflash2 ==="
ls -1 /mnt/c/Users/User/.cache/huggingface/hub 2>/dev/null | grep -iE 'incoai|dflash2' | head -6
echo "--- mixbits GGUF 的 blobs（单目录，只看大小/名字）"
ls -l --time-style=+%m-%d_%H:%M /mnt/c/Users/User/.cache/huggingface/hub/models--mixbits--Qwen3.8-27B-NVFP4-MTP-VL-GGUF/snapshots/*/ 2>/dev/null | head -10
echo
echo "=== 4) 全新 vllm 0.29 安装状态 ==="
[ -d /home/user/vllm029 ] && echo "venv 已建" || echo "venv 未建"
tail -5 $J/dl/vllm029_install.log 2>/dev/null
pgrep -f 'pip install' >/dev/null && echo "pip 在跑" || echo "pip 未在跑"
