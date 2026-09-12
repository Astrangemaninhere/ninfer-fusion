#!/bin/bash
# temp: rebuild with the per-layer probe and find the first NaN layer in Muse's decode stack.
set -u
WIN=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1
WSL=/home/user/ninfer-fusion
f=src/targets/qwen3_6/impl/runtime/text_context_impl.h
cp -f "$WIN/$f" "$WSL/$f" || exit 1
a=$(md5sum "$WIN/$f" | cut -d' ' -f1); b=$(md5sum "$WSL/$f" | cut -d' ' -f1)
[ "$a" = "$b" ] && echo "SYNC_OK" || { echo SYNC_MISMATCH; exit 1; }
cd "$WSL/build" || exit 1
setsid nohup make ninfer -j4 > /home/user/headdbg2_build.log 2>&1 < /dev/null &
for _i in $(seq 1 90); do sleep 10; pgrep -x make >/dev/null || break; done
tail -2 /home/user/headdbg2_build.log

CLI=$WSL/build/apps/ninfer
NINFER_HEADDBG=1 timeout 300 "$CLI" /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 2 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/hd2.out 2>/tmp/hd2.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/hd2.err | head -1)"
echo "== first NAN line (step 1 decode)"
grep -n 'NAN' /tmp/hd2.err | head -3
echo "== layer trace around the first NAN (step 1)"
first=$(grep -n 'NAN' /tmp/hd2.err | head -1 | cut -d: -f1)
if [ -n "$first" ]; then
  start=$((first - 6)); [ $start -lt 1 ] && start=1
  sed -n "${start},$((first + 2))p" /tmp/hd2.err
fi
echo "== total probe lines"; grep -c 'headdbg' /tmp/hd2.err
echo HEADDBG2_DONE
