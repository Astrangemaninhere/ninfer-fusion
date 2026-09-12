#!/bin/bash
# temp: 2-chunk prefill — does the prefill itself see the bad KV at layer 16?
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
MUSE=/home/user/models/muse_glimmer_30b_nvfp4.ninfer

P=$(python3 -c "print('这是一段用于测试多块预填充的中文文本。' * 40, end='')")
echo "prompt chars=${#P}"
NINFER_HEADDBG=1 timeout 400 "$CLI" "$MUSE" --prompt "$P" --max-new 2 --no-thinking \
  --greedy --no-cuda-graph --kv-dtype nvfp4 --max-context 8192 --prefill-chunk 512 \
  --print-token-ids > /tmp/mc.out 2>/tmp/mc.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/mc.err | head -1)"
echo "== first NaN in the whole run (prefill included)"
first=$(grep -n 'NAN' /tmp/mc.err | head -1 | cut -d: -f1)
if [ -n "$first" ]; then
  start=$((first - 4)); [ $start -lt 1 ] && start=1
  sed -n "${start},$((first + 1))p" /tmp/mc.err
else
  echo "no NaN at all"
fi
echo "== L15/L16 prefill stage lines"
grep -E 'L1[56]_(in_x|attn_out)' /tmp/mc.err | head -8
echo MULTICHUNK_DONE
