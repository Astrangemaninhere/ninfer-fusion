#!/bin/bash
# §104 step 1: dump layer 15/16 KV planes + the source BF16 K/V after a Muse prefill.
# usage: _kvforensics.sh [nvfp4|bf16|int8]
set -u
WSL=/home/user/ninfer-fusion
CLI=$WSL/build/apps/ninfer
DUMP=/home/user/bench/kvdump
KV="${1:-nvfp4}"
rm -rf "$DUMP"; mkdir -p "$DUMP"
NINFER_KVDUMP_DIR=$DUMP NINFER_KVDUMP_KV=15,16 NINFER_KVDUMP_LAYERS=15,16 \
  timeout 300 "$CLI" /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 1 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype "$KV" --max-context 4096 --print-token-ids > /tmp/kv_$KV.out 2>/tmp/kv_$KV.err
rc=$?
echo "kv=$KV rc=$rc ids=$(grep -oE 'generated ids.*' /tmp/kv_$KV.err | head -1)"
ls -l "$DUMP" | head -40
python3 /mnt/c/Users/User/Documents/ziqinzhang/_kvdump_analyze.py "$DUMP" | head -120
echo "KVFORENSICS_DONE $KV"
