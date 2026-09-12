#!/bin/bash
# 重装 vllm 0.29（python3.12 venv），换清华镜像、全量输出定位真因
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
W=$J/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
V=/home/user/vllm029
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=$J/dl/vllm029_install2.log
exec > >(tee -a "$LOG") 2>&1
echo "================================================================"
echo "=== vllm 0.29 重装 $(date '+%F %H:%M:%S') ==="

echo "--- 0) venv 与 pip 现状 ---"
[ -x $V/bin/python ] || { python3.12 -m venv "$V" || exit 2; }
VP=$V/bin/python
$VP -V
$VP -m pip -V
$VP -c "import sys; print('prefix', sys.prefix)"

echo "--- 1) 升级 pip/setuptools（镜像） ---"
$VP -m pip install -i $IDX -U pip setuptools wheel 2>&1 | tail -6
$VP -m pip -V

echo "--- 2) 装 numpy 探针（验证索引可用） ---"
$VP -m pip install -i $IDX numpy 2>&1 | tail -5

echo "--- 3) 装 vllm 0.29（镜像 + 本地轮子） ---"
$VP -m pip install -i $IDX --extra-index-url $IDX "$W" 2>&1 | tail -25
echo "pip rc=${PIPESTATUS[0]}"

echo "--- 4) 验证 ---"
$VP -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3
$VP -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a)" 2>&1 | tail -2
$VP -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -2
echo VLLM029_INSTALL2_DONE
