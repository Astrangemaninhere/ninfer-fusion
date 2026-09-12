#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
M=/home/user/models/qwen3_8_27b_nvfp4.ninfer
D2=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:800])
")
echo "=== DFlash2 WITH graph (default) ==="
timeout 300 ./build/apps/ninfer "$M" --draft "$D2" --prompt "$PROMPT" --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 --max-new 300 --no-thinking >/dev/null 2>/tmp/sp1.err
grep -E "decode speed|accept|draft" /tmp/sp1.err | head -3
echo "=== DFlash2 no-cuda-graph ==="
timeout 300 ./build/apps/ninfer "$M" --draft "$D2" --prompt "$PROMPT" --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 --max-new 300 --no-thinking --no-cuda-graph >/dev/null 2>/tmp/sp2.err
grep -E "decode speed|accept|draft" /tmp/sp2.err | head -3
