#!/bin/bash
# 按用户要求：把干净环境 vllm 更新到最新版（0.29.0），留好回滚快照
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
PY="/mnt/c/vllm/venv-clean/Scripts/python.exe"
LOG=$J/dl/vllm_update.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 更新 venv-clean 的 vllm  $(date '+%F %H:%M:%S') ==="

echo "--- 0) 回滚快照（关键包版本 + wheel 列表） ---"
"$PY" -m pip list --format=freeze 2>/dev/null | grep -iE '^(vllm|torch|transformers|flashinfer|xformers|triton|numpy|tilelang|humming)' | tee $J/dl/venv_clean_before.txt
echo "--- 端口占用检查（铁律②：先看是谁） ---"
netstat -ano -p TCP 2>/dev/null | grep -E ':812[0-9]\s' | head -5 || echo "  812x 无占用"

echo "--- 1) 更新到 0.29.0（仅 vllm，不动 torch） ---"
"$PY" -m pip install -U "vllm==0.29.0" 2>&1 | tail -25
echo "pip rc=${PIPESTATUS[0]}"

echo "--- 2) 更新后版本与方法名 ---"
"$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2
"$PY" -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash' in a, 'dflash2' in a, 'dspark' in a)" 2>&1 | tail -2
"$PY" -m pip list --format=freeze 2>/dev/null | grep -iE '^(vllm|torch|transformers)' | tee $J/dl/venv_clean_after.txt
echo VLLM_UPDATE_DONE
