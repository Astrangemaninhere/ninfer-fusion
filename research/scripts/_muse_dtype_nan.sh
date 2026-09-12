#!/bin/bash
# temp: does the Muse layer-16 NaN depend on the KV dtype?
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
MUSE=/home/user/models/muse_glimmer_30b_nvfp4.ninfer

for dtype in nvfp4 bf16 int8; do
  NINFER_HEADDBG=1 timeout 300 "$CLI" "$MUSE" --prompt '1, 2, 3, 4, 5, 6,' --max-new 2 \
    --no-thinking --greedy --no-cuda-graph --kv-dtype "$dtype" --max-context 4096 \
    --print-token-ids > /tmp/hd_$dtype.out 2>/tmp/hd_$dtype.err
  first=$(grep -n 'NAN' /tmp/hd_$dtype.err | head -1 | cut -d: -f1)
  label=$(grep 'NAN' /tmp/hd_$dtype.err | head -1 | grep -oE 'L[0-9]+_[a-z]+')
  echo "[$dtype] rc=$? first_nan_line=$first first_nan_label=${label:-none} ids=$(grep -oE 'generated ids.*' /tmp/hd_$dtype.err | head -1)"
done
echo MUSE_DTYPE_NAN_DONE
