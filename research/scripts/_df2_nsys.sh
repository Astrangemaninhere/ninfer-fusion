#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
MDF=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:800])
")
timeout 400 nsys profile -o /home/user/bench/df2_graph --trace=cuda --cuda-memory-usage=false \
  --cuda-graph-trace=node --force-overwrite true \
  ./build/apps/ninfer "$MDF" --prompt "$PROMPT" \
  --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 4096 --kv-capacity 4096 \
  --spec dflash2 --draft-tokens 7 --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-new 60 --no-thinking >/dev/null 2>/tmp/df2_nsys.log
echo NSYS_EXIT=$?
cd /home/user/bench
nsys stats --report cuda_gpu_trace --format csv df2_graph.nsys-rep > /home/user/bench/trace_df2.csv 2>/dev/null
echo STATS_DONE
grep -E "decode speed|acceptance" /tmp/df2_nsys.log
