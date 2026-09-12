#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
M=/home/user/models/qwen3_8_27b_nvfp4.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:1500])
")
echo "=== dense attempt: no --spec flag at all ==="
timeout 300 ./build/apps/ninfer "$M" --prompt "$PROMPT" --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 --max-new 300 --no-thinking --no-cuda-graph 2>&1 | tail -20
echo ALL_DONE
