#!/bin/bash
# (a) 回滚到已知可用的本地轮子 0.26.0+cu132；(b) 查 1Cat-vLLM fork 是否含 dflash2 及构建配方
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
PY="/mnt/c/vllm/venv-clean/Scripts/python.exe"
W=$J/dl/vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl
LOG=$J/dl/vllm_rollback.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 回滚 $(date '+%H:%M:%S') ==="
ls -l "$W" 2>/dev/null | tail -1
"$PY" -m pip install --force-reinstall --no-deps "$W" 2>&1 | tail -8
echo "pip rc=${PIPESTATUS[0]}"
echo "--- 验证导入 ---"
"$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3
"$PY" -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a)" 2>&1 | tail -2
echo
echo "=== 1Cat-vLLM fork：版本与 dflash2 是否存在（单文件） ==="
grep -m3 -E 'version|__version__' $J/1Cat-vLLM/vllm/version.py 2>/dev/null || grep -m3 -E 'version =' $J/1Cat-vLLM/vllm/__init__.py 2>/dev/null | head -3
ls -1 $J/1Cat-vLLM/vllm/model_executor/models/qwen3_dflash2.py $J/1Cat-vLLM/vllm/model_executor/models/qwen3_dflash.py 2>/dev/null
echo "--- fork 里 dflash2 方法名（单文件）"
grep -nE 'dflash2|dflash' $J/1Cat-vLLM/vllm/config/speculative.py 2>/dev/null | head -8
echo
echo "=== 构建 vllm 轮子的脚本（在工作区根，单目录 ls 过滤） ==="
ls -1 $J/*.bat $J/*.ps1 2>/dev/null | grep -iE 'build|vllm|wheel' | head -12
echo ROLLBACK_AND_CHECK_DONE
