#!/bin/bash
# Inspect the vllm 0.29.0 manylinux wheel for dflash2 support WITHOUT installing.
W=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
echo "=== wheel ==="
ls -la "$W"
echo "=== sha256 ==="
sha256sum "$W"
echo "=== total entries ==="
unzip -l "$W" 2>/dev/null | tail -2
echo "=== DFASH2 entries in wheel ==="
unzip -l "$W" 2>/dev/null | grep -i 'dflash2' | head -40
echo "--- dflash2 count ---"
unzip -l "$W" 2>/dev/null | grep -ic 'dflash2'
echo "=== DFASH (any) entries ==="
unzip -l "$W" 2>/dev/null | grep -i 'dflash' | head -40
echo "--- dflash count ---"
unzip -l "$W" 2>/dev/null | grep -ic 'dflash'
echo "=== DSPARK entries ==="
unzip -l "$W" 2>/dev/null | grep -i 'dspark' | head -20
echo "--- dspark count ---"
unzip -l "$W" 2>/dev/null | grep -ic 'dspark'
echo "=== spec_decode dir listing ==="
unzip -l "$W" 2>/dev/null | grep -i 'spec_decode' | head -50
echo "=== qwen3 model files ==="
unzip -l "$W" 2>/dev/null | grep -iE 'models/qwen3' | head -40
