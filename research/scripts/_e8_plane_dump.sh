#!/bin/bash
# §96: dump the e8 KV planes of layers 13/14/15 after a long prefill and look
# for plane aliasing / corruption. usage: _e8_plane_dump.sh [chars] [layerspec]
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
DUMP=/home/user/bench/kvdump_e8
CHARS="${1:-100000}"
LAYERS="${2:-14-15:e8}"
rm -rf "$DUMP"; mkdir -p "$DUMP"
PROMPT=$(python3 -c "
s = open('/home/user/spec_prompt_base.txt', encoding='utf-8', errors='ignore').read()
print(s[:$CHARS])
")
NINFER_KVDUMP_DIR=$DUMP NINFER_KVDUMP_LAYERS=13,14,15 NINFER_KVDUMP_PAGES=64 \
  timeout 900 "$CLI" /home/user/models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "$PROMPT" --max-new 1 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --kv-layer-storage "$LAYERS" \
  --max-context 32768 --kv-capacity 32768 --print-token-ids > /tmp/e8.out 2>/tmp/e8.err
echo "rc=$? layers=$LAYERS chars=$CHARS ids=$(grep -oE 'generated ids.*' /tmp/e8.err | head -1)"
ls -l "$DUMP" | head -25
python3 /mnt/c/Users/User/Documents/ziqinzhang/_e8_alias_check.py "$DUMP"
echo E8_PLANE_DUMP_DONE
