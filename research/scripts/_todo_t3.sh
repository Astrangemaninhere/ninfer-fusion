#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
D=$Z/_collab/build/T3_src
echo "=== _TODO.md entries around 126 / target-shift / ids16 ==="
grep -n 'target-shift\|target_shift\|ids16\|shift 0\|shift=0\|shift 1\|shift=1' $Z/_TODO.md 2>/dev/null | head -40
echo
echo "=== _TODO.md size ==="
wc -l $Z/_TODO.md
echo
echo "=== dflash2 model dir ==="
ls -la $Z/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/ 2>/dev/null
echo
echo "=== provenance logs: export ==="
ls -la $Z/dl/ 2>/dev/null | grep -i 'export\|df2\|dflash' | head -20
echo
echo "=== _collab reports: newest 25 by mtime ==="
ls -lat $Z/_collab/*.md $Z/_collab/*.diff 2>/dev/null | head -25
