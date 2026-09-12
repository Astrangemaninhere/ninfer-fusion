#!/bin/bash
# plain vs spec 的 KV dump + 对照（目录落 dl/，不用 /tmp）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
A=$J/dl/kvA
B=$J/dl/kvB
rm -rf "$A" "$B"; mkdir -p "$A" "$B"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LAYERS=0,15
cd "$R/build" || exit 3

echo "=== plain 臂 dump（层 $LAYERS；KV dump 需 --no-cuda-graph） ==="
NINFER_KVDUMP_DIR="$A" NINFER_KVDUMP_KV="$LAYERS" timeout 900 ./apps/ninfer \
  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --prompt "$P" --max-new 16 \
  --max-context 4096 --no-thinking --greedy --no-cuda-graph > "$J/dl/kvdump_plain.log" 2>&1
echo "  rc=$? files=$(ls "$A" | wc -l)"

echo "=== spec 臂 dump ==="
NINFER_KVDUMP_DIR="$B" NINFER_KVDUMP_KV="$LAYERS" timeout 900 ./apps/ninfer \
  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --prompt "$P" --max-new 16 \
  --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph > "$J/dl/kvdump_spec.log" 2>&1
echo "  rc=$? files=$(ls "$B" | wc -l)"

echo
echo "=== 对照 ==="
python3 "$J/_kv_cmp.py" "$A" "$B" 2>&1 | tail -16
