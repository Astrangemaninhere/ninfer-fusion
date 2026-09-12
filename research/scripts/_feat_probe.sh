#!/bin/bash
# Feature-link probe: does DFlash2 draft actually consume target features?
# Experiment: if we zero the context append (features never enter the cyclic
# cache), acceptance should collapse if features matter. Instead, we test the
# reverse indirectly: first-round (prefill-only, no verify history) pos1
# acceptance vs later rounds. A correct feature link gives high pos1 from
# round 1 (features from prefill are in cache).
# We already have per-round d2dbg output; this run captures it at 1.5K.
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
MDF=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:1500])
")
timeout 200 ./build/apps/ninfer "$MDF" --prompt "$PROMPT" --kv-dtype bf16 --kv-layer-storage all:bf16 \
  --max-context 4096 --kv-capacity 4096 \
  --spec dflash2 --draft-tokens 7 --greedy --max-new 96 --no-thinking --no-cuda-graph 2>&1 \
  | grep -aE 'd2dbg|acceptance rate|accepted by pos|rounds' | head -50
echo ALL_DONE
