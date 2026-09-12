#!/bin/bash
F=/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/text_context_impl.h
echo "=== 1) target_verify_batch 定义位置 ==="
grep -n "target_verify_batch" "$F" | head -6
echo
echo "=== 2) rope_delta / offset_i32_positions 的所有使用点 ==="
grep -n "rope_delta\|offset_i32_positions" "$F" | head -12
echo
echo "=== 3) 文本注意力的 positions/rope 入口（gqa_attention 调用点） ==="
grep -n "gqa_attention\|ops::rope" "$F" | head -16
echo
echo "=== 4) active_* 绑定（verify/mtp 批次绑定的参数） ==="
grep -n "active_valid_columns_\|active_sequence_width_\|active_sequence_batch_\|active_backend_kv_table_rows_\|active_rope_delta_" "$F" | head -20
