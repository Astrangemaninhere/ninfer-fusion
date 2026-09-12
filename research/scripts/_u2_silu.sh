#!/bin/bash
# U2: who consumes ops::silu / math.cuh silu + S51 landing time
set -u
R=/home/user/ninfer-fusion
cd "$R" || exit 1
echo "### users of silu( / sigmoid( (device helpers)"
grep -rn '\bsilu(\|\bsigmoid(' src/ --include=*.cuh --include=*.cu --include=*.cpp 2>/dev/null | grep -v 'math.cuh:' | sed "s#$R/##" | head -25
echo
echo "### does bf16 draft path use silu? (linear_swiglu bf16)"
grep -rn 'silu' src/ops/linear_swiglu/bf16/ 2>/dev/null | head -8
echo
echo "### math.h (non-cuda) silu decl"
grep -rn 'silu' src/ops/common/math.h 2>/dev/null | head -8
echo
echo "### S51 landing evidence in _collab"
grep -rn 'S51' /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/E_s51_and_geometry_report.md 2>/dev/null | head -12
echo
echo "### E8_s51 doc head"
head -18 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E8_s51_nvfp4_silu.md 2>/dev/null
echo
echo "### git? (is there a repo or snapshot?)"
ls -d "$R/.git" 2>/dev/null || echo "  no .git"
ls /home/user/ninfer-fusion/_orig_quarantine 2>/dev/null | head -5
echo
echo "### nvcc/make running now?"
pgrep -af 'nvcc|cc1plus' 2>/dev/null | head -3 || echo "  none"
