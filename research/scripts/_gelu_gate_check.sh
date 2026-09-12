#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
B=/home/user/ninfer-fusion
echo "=== is there any GELU activation op? ==="
grep -rliE 'gelu' "$R/include/ninfer/ops" "$R/src/ops" 2>/dev/null | head -8
grep -rniE 'gelu' "$R/include/ninfer/ops/swiglu.h" "$R/include/ninfer/ops/activation.h" 2>/dev/null | head -5
ls "$R/include/ninfer/ops/" 2>/dev/null | grep -iE 'act|gelu|silu|mlp' | head -10
echo
echo "=== who uses sigmoid_gate_mul / the attn input-proj output gate? ==="
grep -rn 'sigmoid_gate_mul' "$R/src" --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | grep -v CMakeLists | head -8 | cut -c1-140
grep -rn 'output_gate' "$R/src/targets" "$R/src/ops/wrapper" 2>/dev/null | head -8 | cut -c1-140
echo
echo "=== target config knobs: is activation selectable? ==="
grep -rnE 'constexpr|static_assert' "$R/src/targets/qwen3_6_27b/impl/config.h" | grep -iE 'act|silu|gelu|mlp' | head -8 | cut -c1-130
echo
echo "=== adapt.py: the detector block I need to extend (lines 40-180) ==="
sed -n '40,80p' "$R/tools/archkit/adapt.py" | cat -n | sed 's/^/  /' | cut -c1-150
echo "  ..."
sed -n '126,180p' "$R/tools/archkit/adapt.py" | cat -n | sed 's/^/  /' | cut -c1-150
