#!/bin/bash
# Window G: the WSL VM itself died (E_UNEXPECTED) after F started. All three
# build deaths this morning share one cause: the WSL ceiling in .wslconfig was
# 24GB while a single decode-TU compile peaks above it (cicc + ptxas in one
# nvcc). Mitigations now in place:
#   * .wslconfig memory 24GB -> 26GB, swap 16GB -> 32GB (takes effect on this boot)
#   * NVCC_PREPEND_FLAGS=--split-compile-extended=8  (device compilation split,
#     so ptxas peaks scale with a partition instead of the whole TU)
#   * -j1 + up to 4 retries: a kill costs the in-flight TU, never the window.
# Pipeline unchanged: relink proof -> assertive Muse check -> assertive e8 probe
# -> resume W9 training.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/window_g.log
: > "$LOG"
exec >> "$LOG" 2>&1
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
echo "=== window G start $(date +%F' '%H:%M:%S) ==="
echo "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"
free -g | head -2
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
pgrep -af 'make|nvcc|ptxas|train_dflash2' | grep -v $$ || true

cd /home/user/ninfer-fusion/build || { echo WINDG_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")

rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o

rc=1
for attempt in 1 2 3 4; do
  echo "--- make ninfer -j1 attempt $attempt $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  make ninfer -j1 > /tmp/wg_make.log 2>&1
  rc=$?
  tail -6 /tmp/wg_make.log
  [ $rc -eq 0 ] && break
  echo "attempt $attempt failed rc=$rc; retry (make resumes from built .o)"
  sleep 5
done
if [ $rc -ne 0 ]; then
  echo "MAKE_FAILED rc=$rc — 不做验证, 不让陈旧二进制产出绿色结果"
  echo WINDG_FAIL
  exit 3
fi
after=$(stat -c %Y "$BIN")
ls -l --time-style=+%H:%M:%S "$BIN"
if [ "$after" -le "$before" ]; then
  echo "RELINK_MISSING ($before -> $after) — 不验证"
  echo WINDG_FAIL
  exit 3
fi
echo "REBUILT OK ($before -> $after)"

echo "--- [1/3] muse verification $(date +%H:%M:%S) ---"
bash "$J/_muse_verify.sh"
echo "--- [2/3] e8 post-fix probe $(date +%H:%M:%S) ---"
bash "$J/_e8_postfix.sh"
echo "--- [3/3] resume W9 training $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDG_FAIL; exit 3; }
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 30
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "=== window G done $(date +%F' '%H:%M:%S) ==="
echo WINDG_DONE
