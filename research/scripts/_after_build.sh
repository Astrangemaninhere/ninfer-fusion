#!/bin/bash
# Wait for the running build, then (once the training has saved step_000500 so a
# pause costs nothing) run the GPU window: Muse row-scale verification + e8
# source dump, then resume the training from the checkpoint.
set -u
while pgrep -x make >/dev/null; do sleep 15; done
echo "build finished at $(date +%H:%M:%S)"
for _i in $(seq 1 120); do
  if [ -f /mnt/c/Users/User/Documents/ziqinzhang/data/dflash2_ckpts/step_000500.pt ]; then
    echo "checkpoint step_000500 present at $(date +%H:%M:%S)"
    break
  fi
  sleep 20
done
bash /mnt/c/Users/User/Documents/ziqinzhang/_gpu_window.sh
echo AFTER_BUILD_DONE
