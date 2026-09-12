#!/bin/bash
# Rebuild with the row-scale bounds fix and re-check Muse's NaN/word-salad.
set -u
echo "=== sync ==="
bash /mnt/c/Users/User/Documents/ziqinzhang/_sync.sh
echo "=== build ==="
cd /home/user/ninfer-fusion/build || exit 1
make ninfer -j12 2>&1 | tail -4
echo "=== muse recheck ==="
bash /mnt/c/Users/User/Documents/ziqinzhang/_muse_recheck.sh
echo "=== muse text sample ==="
timeout 300 /home/user/ninfer-fusion/build/apps/ninfer \
  /home/user/models/muse_glimmer_30b_nvfp4.ninfer \
  --prompt '用一句话介绍杭州。' --max-new 48 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 2>&1 | tail -12
echo ROWSCALE_FIX_DONE
