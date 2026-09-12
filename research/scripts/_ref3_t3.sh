#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
tr -d '\r' < $Z/tmp/vllm-qwen3_dflash2.py > $D/vllm_qwen3_dflash2.py
tr -d '\r' < $Z/train_dspark.py > $D/train_dspark.py
tr -d '\r' < $Z/dl/aeon-dflash.py > $D/aeon_dflash.py
tr -d '\r' < $Z/_collab/E9_s52_dflash2_k_slice.diff > $D/E9_s52_k_slice.diff
tr -d '\r' < $Z/_collab/E9_s52_dflash2_k_slice.md  > $D/E9_s52_k_slice.md
wc -l $D/vllm_qwen3_dflash2.py $D/train_dspark.py $D/aeon_dflash.py $D/E9_s52_k_slice.md
echo
echo "=== vllm ref: block/column/draft logic ==="
grep -n 'block\|columns\|:\]\|logits\|mask\|position\|draft' $D/vllm_qwen3_dflash2.py | head -60
