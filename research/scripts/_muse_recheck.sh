#!/bin/bash
# Re-check Muse's per-layer NaN state on the current build (no rebuild).
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
NINFER_HEADDBG=1 timeout 300 "$CLI" /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 2 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/hd.out 2>/tmp/hd.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/hd.err | head -1)"
echo "== first NaN-bearing layer probes =="
grep -nE 'L[0-9]+_(attn|gdn|mlp)|post_embed|post_layers_x|final_hidden|logits' /tmp/hd.err | grep -n 'NAN' | head -12
echo "== L15/L16 rows =="
grep -E 'L1[456]_(attn|gdn|mlp)|L1[56]_(in_x|q|k|v|qn|kn|attn_out|out_x)' /tmp/hd.err | head -30
echo MUSE_NAN_RECHECK_DONE
