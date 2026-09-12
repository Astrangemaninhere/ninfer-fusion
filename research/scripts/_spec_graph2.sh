#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
MDF=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:800])
")
echo "=== DFlash2 WITH graph (default) ==="
timeout 400 ./build/apps/ninfer "$MDF" --prompt "$PROMPT" \
  --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 \
  --spec dflash2 --draft-tokens 7 --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-new 300 --no-thinking >/dev/null 2>/tmp/spg1.err
grep -E "decode speed|acceptance rate" /tmp/spg1.err | head -3
echo "=== DFlash2 no-cuda-graph ==="
timeout 400 ./build/apps/ninfer "$MDF" --prompt "$PROMPT" \
  --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 \
  --spec dflash2 --draft-tokens 7 --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-new 300 --no-thinking --no-cuda-graph >/dev/null 2>/tmp/spg2.err
grep -E "decode speed|acceptance rate" /tmp/spg2.err | head -3
