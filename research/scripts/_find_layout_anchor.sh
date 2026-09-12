#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
echo "=== layouts_impl.h 里 spec 聚合与相关函数的锚点 ==="
grep -nE 'layer_residual|layer_sliding_windows|DecoderStateSpec|DecoderStateLayout|plan_decoder_state|sliding' "$F" | head -20 | cut -c1-140
echo
echo "=== 该聚合的上下文（前后 12 行） ==="
L=$(grep -n 'layer_residual = ' "$F" | head -1 | cut -d: -f1)
echo "  anchor line: $L"
sed -n "$((L-12)),$((L+4))p" "$F" | cat -n | sed "s/^/  /" | cut -c1-140
