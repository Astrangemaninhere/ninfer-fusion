#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "=== engine rope op: is there a partial/rotary_dim parameter? ==="
sed -n '1,40p' "$R/include/ninfer/ops/rope.h" 2>/dev/null | cut -c1-120
echo
echo "=== per-layer / per-type rope theta support in a target config ==="
grep -rnE 'rope_theta|layer_rope_theta|rotary|partial_rot' "$R/src/targets/qwen3_6_27b/impl/config.h" 2>/dev/null | head -12 | cut -c1-130
grep -rn 'rope_theta' "$R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h" 2>/dev/null | head -6 | cut -c1-120
echo
echo "=== attention output gate (headwise g_proj) anywhere in the engine? ==="
grep -rniE 'output_gate|attn_output_gate|headwise|g_proj|sigmoid_gate' "$R/src" "$R/include" 2>/dev/null | head -10 | cut -c1-140
echo
echo "=== MLP activation: is silu hardwired or is there an act knob? ==="
grep -rniE 'activation|act_kind|gelu|swiglu' "$R/src/targets/qwen3_6_27b/impl/config.h" "$R/include/ninfer/ops/silu_mul.h" 2>/dev/null | head -10 | cut -c1-130
echo
echo "=== tied head support (lm_head absent => embed^T)? ==="
grep -rniE 'tie_word|tied|lm_head.*embed|embed.*lm_head' "$R/src/targets/qwen3_6_27b/impl/" "$R/tools/convert/qwen3_6_27b/" 2>/dev/null | head -10 | cut -c1-140
