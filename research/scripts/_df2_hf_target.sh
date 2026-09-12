#!/bin/bash
# 定点：0.29 的 SpeculativeMethod 是否含 dflash2；qwen3_dflash2 期望的 config/权重键
PY="/mnt/c/vllm/venv-clean/Scripts/python.exe"
P=/mnt/c/vllm/venv-clean/Lib/site-packages/vllm
echo "=== 1) 0.29 允许的 spec 方法名 ==="
"$PY" -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a); print([x for x in a if 'dflash' in x or 'dspark' in x])" 2>&1 | tail -3
echo
echo "=== 2) qwen3_dflash2.py 的架构与配置字段 ==="
grep -nE 'class |architectures|config\.|num_hidden_layers|mask|conv|context_key|hidden_norm' $P/model_executor/models/qwen3_dflash2.py 2>/dev/null | head -30
echo
echo "=== 3) 该文件的权重加载：期望的键名形态 ==="
grep -nE 'load_weights|weight_loader|named_parameters|def _|self\.' $P/model_executor/models/qwen3_dflash2.py 2>/dev/null | head -25
