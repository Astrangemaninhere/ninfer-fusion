#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
M=/home/user/models/qwen3_8_27b_nvfp4.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:1500])
")
timeout 400 /usr/local/bin/nsys profile --stats=true --force-overwrite true -o /tmp/nsys_dense \
  ./build/apps/ninfer "$M" --prompt "$PROMPT" --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 8192 --kv-capacity 8192 --max-new 60 --no-thinking --no-cuda-graph \
  > /tmp/nsys_run.log 2>&1
echo "NSYS exit: $?"
echo "=== GPU Kernel Summary (time) ==="
/usr/local/bin/nsys stats --report cuda_gpu_kern_sum --format table /tmp/nsys_dense.nsys-rep 2>/dev/null | head -30
echo NSYS_DONE
