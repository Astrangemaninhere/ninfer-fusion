#!/bin/bash
# 关键判定：老 venv 的 vllm 是否含 fork 特有的 dflash2 支持
V=/mnt/c/vllm/venv/Lib/site-packages/vllm
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 老 venv 是否有 qwen3_dflash2.py ==="
ls -l --time-style=+%m-%d_%H:%M $V/model_executor/models/qwen3_dflash2.py 2>/dev/null || echo "  无（则它不支持 dflash2）"
echo
echo "=== 2) 老 venv 的 speculative.py 是否含 fork 特有名字 ==="
grep -cE 'dflash_ddtree|_get_dflash2_checkpoint_draft_tokens|DDTreeBuildMode' $V/config/speculative.py 2>/dev/null
grep -nE 'dflash_ddtree|DDTreeBuildMode|dflash2' $V/config/speculative.py 2>/dev/null | head -6
echo
echo "=== 3) 老 venv 的 vllm 版本与是否有 dflash2 模型 ==="
grep -m2 -E '__version__' $V/version.py 2>/dev/null | head -3
ls -1 $V/model_executor/models 2>/dev/null | grep -iE 'dflash|dspark' | head
echo
echo "=== 4) 该 venv 能否 import vllm（实证） ==="
/mnt/c/vllm/venv/Scripts/python.exe -c "import vllm; print('OK', vllm.__version__)" 2>&1 | tail -3
