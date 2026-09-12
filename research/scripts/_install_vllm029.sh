#!/bin/bash
# 重装一个全新的 vllm 0.29 环境（WSL/Linux），用本地已下好的 manylinux 轮子
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
W=$J/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
V=/home/user/vllm029
LOG=$J/dl/vllm029_install.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 全新 vllm 0.29 环境安装 $(date '+%F %H:%M:%S') ==="
ls -l "$W" | tail -1
echo "--- 磁盘 ---"; df -h /home | tail -1

echo "--- 1) 建 venv ---"
python3 -m venv "$V" || { echo VENV_FAIL; exit 2; }
V_PY=$V/bin/python
$V_PY -m pip install -q -U pip setuptools wheel 2>&1 | tail -3
$V_PY -V

echo "--- 2) 装 vllm 0.29（让 pip 解析依赖，网络下载） ---"
$V_PY -m pip install "$W" 2>&1 | tail -18
echo "pip rc=${PIPESTATUS[0]}"

echo "--- 3) 验证导入 + dflash2 支持 ---"
$V_PY -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3
$V_PY -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a)" 2>&1 | tail -2
echo "--- 4) 构建扩展是否存在 ---"
ls -1 $V/lib/python3*/site-packages/vllm/*.so 2>/dev/null | head -6
$V_PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -2
echo VLLM029_INSTALL_DONE
