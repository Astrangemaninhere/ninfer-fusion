#!/bin/bash
# 隔离后的第一步：降级 flashinfer-python 到与 cubin 匹配的 0.6.13（--no-deps 以免动 torch）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=$J/dl/fi_downgrade.log
exec > >(tee -a "$LOG") 2>&1
echo "=== flashinfer 降级 $(date '+%H:%M:%S') ==="
free -g | head -2
echo "--- 当前 ---"
$V -m pip list 2>/dev/null | grep -i flashinfer
echo "--- 降级 flashinfer-python -> 0.6.13 (--no-deps) ---"
$V -m pip install -i $IDX --no-deps "flashinfer-python==0.6.13" 2>&1 | tail -6
echo "pip rc=${PIPESTATUS[0]}"
echo "--- 之后 ---"
$V -m pip list 2>/dev/null | grep -i flashinfer
echo "--- 版本是否已对齐（不再需要 bypass） ---"
$V -c "import flashinfer; print('flashinfer import ok:', flashinfer.__version__)" 2>&1 | tail -3
echo "--- 去掉 bypass 再试一次导入 ---"
env -u FLASHINFER_DISABLE_VERSION_CHECK $V -c "import flashinfer; print('no-bypass import ok:', flashinfer.__version__)" 2>&1 | tail -3
echo "--- torch 是否被动过 ---"
$V -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -1
echo FI_DOWNGRADE_DONE
