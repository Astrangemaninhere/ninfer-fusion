#!/bin/bash
# temp: build with the head probe and dump the Muse decode head tensors.
set -u
WIN=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1
WSL=/home/user/ninfer-fusion
f=src/targets/qwen3_6/impl/runtime/text_context_impl.h
cp -f "$WIN/$f" "$WSL/$f" || exit 1
a=$(md5sum "$WIN/$f" | cut -d' ' -f1); b=$(md5sum "$WSL/$f" | cut -d' ' -f1)
[ "$a" = "$b" ] && echo "SYNC_OK" || { echo SYNC_MISMATCH; exit 1; }
cd "$WSL/build" || exit 1
setsid nohup make ninfer -j4 > /home/user/headdbg_build.log 2>&1 < /dev/null &
for _i in $(seq 1 90); do sleep 10; pgrep -x make >/dev/null || break; done
tail -3 /home/user/headdbg_build.log

CLI=$WSL/build/apps/ninfer
echo "== Muse decode head probe"
NINFER_HEADDBG=1 timeout 300 "$CLI" /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 4 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/hd_muse.out 2>/tmp/hd_muse.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/hd_muse.err | head -1)"
grep '\[headdbg\]' /tmp/hd_muse.err | head -12
echo "== qwen control"
NINFER_HEADDBG=1 timeout 300 "$CLI" /home/user/models/qwen3_8_27b_nvfp4.ninfer \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 4 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/hd_qwen.out 2>/tmp/hd_qwen.err
echo "rc=$? ids=$(grep -oE 'generated ids.*' /tmp/hd_qwen.err | head -1)"
grep '\[headdbg\]' /tmp/hd_qwen.err | head -6
echo HEADDBG_DONE
