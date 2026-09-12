#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "=== E8 probe: the decisive numbers ==="
grep -nE 'zero|largest|identical|max (abs|rel)|table|x =|bf16' "$J/E8_s51_silu_probe.txt" 2>/dev/null | head -24 | cut -c1-140
echo
echo "=== E8 report: headline + reachability + family sweep ==="
grep -nE '^#|^##|结论|致命|同族|reachab|family|sites|文件|dry-run' "$J/E8_s51_nvfp4_silu.md" 2>/dev/null | head -22 | cut -c1-130
echo
echo "=== E8 diff stat ==="
grep -cE '^\+' "$J/E8_s51_nvfp4_silu.diff" 2>/dev/null | sed 's/^/  added lines: /'
grep -cE '^-' "$J/E8_s51_nvfp4_silu.diff" 2>/dev/null | sed 's/^/  removed lines: /'
grep -E '^\+\+\+|^---' "$J/E8_s51_nvfp4_silu.diff" 2>/dev/null | head -8
echo
echo "=== which TU compiles that header? is it in this build's remaining plan? ==="
grep -rln 'nvfp4_linear_swiglu_w4a4_tma.cuh' /home/user/ninfer-fusion/src --include=*.cu --include=*.cuh 2>/dev/null | head -5
grep -oE 'ops/linear_swiglu[^ ]*' /tmp/pa_make_1.log 2>/dev/null | head -3
