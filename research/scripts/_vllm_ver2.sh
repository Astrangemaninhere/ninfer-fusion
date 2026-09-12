#!/bin/bash
# 快查（都是定点、快速）：允许的 spec 方法名 + 可用最新版 vllm
PY="/mnt/c/vllm/venv-clean/Scripts/python.exe"
echo "=== 1) 干净环境实际允许的 SpeculativeMethod ==="
"$PY" -c "import typing,sys; from vllm.config.speculative import SpeculativeMethod; print('ALLOWED =', typing.get_args(SpeculativeMethod))" 2>&1 | tail -3
echo
echo "=== 2) 已装版本 ==="
"$PY" -c "import vllm,sys; print('vllm', vllm.__version__, 'py', sys.version.split()[0])" 2>&1 | tail -2
echo
echo "=== 3) 可用的 vllm 版本（pip index，需网络） ==="
timeout 180 "$PY" -m pip index versions vllm 2>&1 | head -5
echo
echo "=== 4) 干净环境里 dflash 的实现要点（单文件） ==="
sed -n '1,40p' /mnt/c/vllm/venv-clean/Lib/site-packages/vllm/v1/spec_decode/dflash.py 2>/dev/null
