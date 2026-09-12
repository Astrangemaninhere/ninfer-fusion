#!/bin/bash
# temp: rebuild with the attn_mix stage probes and find where Muse's divergence starts.
set -u
WIN=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1
WSL=/home/user/ninfer-fusion
f=src/targets/qwen3_6/impl/runtime/text_context_impl.h
cp -f "$WIN/$f" "$WSL/$f" || exit 1
a=$(md5sum "$WIN/$f" | cut -d' ' -f1); b=$(md5sum "$WSL/$f" | cut -d' ' -f1)
[ "$a" = "$b" ] && echo "SYNC_OK" || { echo SYNC_MISMATCH; exit 1; }
cd "$WSL/build" || exit 1
setsid nohup make ninfer -j4 > /home/user/mixprobe_build.log 2>&1 < /dev/null &
for _i in $(seq 1 90); do sleep 10; pgrep -x make >/dev/null || break; done
tail -2 /home/user/mixprobe_build.log

CLI=$WSL/build/apps/ninfer
NINFER_HEADDBG=1 timeout 300 "$CLI" /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 2 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/mx.out 2>/tmp/mx.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/mx.err | head -1)"
echo "== L15/L16 stage probes (decode steps only, skip the prefill block)"
grep -E 'L1[56]_(in_x|q|k|v|qn|kn|attn_out|out_x)' /tmp/mx.err | head -24
echo MIXPROBE_DONE
