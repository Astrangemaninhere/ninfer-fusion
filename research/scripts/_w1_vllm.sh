#!/bin/bash
A=/home/user/ninfer-fusion
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== conv gate line (landed or not?):"
sed -n '20,40p' "$A/src/ops/kernel/dflash2_grouped_conv.cuh"
echo
echo "=== V3 patch target lines (dfile):"
grep -n 'position = t' "$A/src/ops/kernel/dflash2_grouped_conv.cuh"
echo
echo "=== vLLM lane (dfile 58-64 of qwen3_dflash.py):"
sed -n '50,70p' "$W/vllm-venv/Lib/site-packages/vllm/model_executor/models/qwen3_dflash.py"
echo
echo "=== vLLM dflash2 (_grouped_conv + causal hints) from tmp copy:"
grep -n 'is_causal\|_dflash_layer_causal\|causal\|block_size\|sliding' "$W/tmp/vllm-qwen3_dflash2.py" | head -30
echo
echo "=== vLLM speculator sample_off / SAMPLE_FROM_ANCHOR:"
grep -rn 'SAMPLE_FROM_ANCHOR\|sample_off' "$W/vllm-venv/Lib/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py" 2>/dev/null | head
