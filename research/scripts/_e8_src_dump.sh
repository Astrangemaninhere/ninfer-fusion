#!/bin/bash
# §96: dump BOTH the append-source K/V and the stored planes for the e8 layers
# at 32K context, so the quantisation error can be measured per layer/position.
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
DUMP=/home/user/bench/kvdump_e8src
CHARS="${1:-100000}"
rm -rf "$DUMP"; mkdir -p "$DUMP"
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:$CHARS])
")
NINFER_KVDUMP_DIR=$DUMP NINFER_KVDUMP_KV=13,14,15 NINFER_KVDUMP_LAYERS=13,14,15 \
NINFER_KVDUMP_PAGES=64 \
  timeout 900 "$CLI" /home/user/models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "$PROMPT" --max-new 1 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --kv-layer-storage 14-15:e8 \
  --max-context 32768 --kv-capacity 32768 --print-token-ids > /tmp/e8s.out 2>/tmp/e8s.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/e8s.err | head -1)"
ls "$DUMP" | wc -l
python3 /mnt/c/Users/User/Documents/ziqinzhang/_e8_error.py "$DUMP" 13,14,15
echo E8_SRC_DUMP_DONE
