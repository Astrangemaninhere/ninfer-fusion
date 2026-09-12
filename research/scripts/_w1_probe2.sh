#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== build/apps listing:"
ls -la --time-style=long-iso "$A/build/apps/" 2>/dev/null
echo "=== libs:"
find "$A/build" -name '*.so' -o -name 'ninfer' -o -name 'ninfer-serve' 2>/dev/null | head -20
echo "=== _orig_quarantine:"
ls -la --time-style=long-iso "$A/_orig_quarantine/" 2>/dev/null | head -30
echo "=== any .orig/.bak around ops:"
find "$A/src" -name '*.orig' -o -name '*.bak' -o -name '*.rej' 2>/dev/null | head -20
echo "=== grep for causal in ops (swa/bidirectional):"
grep -rn 'causal' "$A/src/ops/kernel/bidirectional_gqa_attention.cuh" "$A/src/ops/launcher/swa.cu" "$A/src/ops/wrapper/swa.cpp" "$A/include/ninfer/ops/swa.h" | head -20
echo "=== windows _collab root files:"
ls -la --time-style=long-iso /mnt/c/Users/User/Documents/ziqinzhang/_collab/*.md /mnt/c/Users/User/Documents/ziqinzhang/_collab/*.diff 2>/dev/null | tail -40
