#!/bin/bash
# 装 runai-model-streamer（真流式加载器，主机峰值≈单张量），然后用它重跑
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=$J/dl/streamer_install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 装 runai-model-streamer $(date '+%H:%M:%S') ==="
$V -m pip install -i $IDX --no-deps runai-model-streamer 2>&1 | tail -6
echo "pip rc=${PIPESTATUS[0]}"
$V -m pip list 2>/dev/null | grep -i runai
echo "--- 导入验证 ---"
$V -c "import runai_model_streamer; print('runai streamer ok')" 2>&1 | tail -3
echo STREAMER_INSTALL_DONE
