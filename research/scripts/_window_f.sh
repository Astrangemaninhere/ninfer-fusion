#!/bin/bash
# Window F: E died 2 minutes in — journalctl shows the kernel OOM killer took a
# ptxas whose *single* process had total-vm 20.4GB, so -j1 alone is not enough:
# one nvcc+ptxas pair for the decode TU already exceeds this WSL's 24GB ceiling.
#
# Fix without touching CMake or sources: nvcc honours NVCC_PREPEND_FLAGS, so the
# whole run gets --split-compile-extended=8 (device compilation partitioned; the
# peak scales with the partition, not the TU). -j2 is then affordable, and the
# retry loop resumes from whatever .o already landed.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/window_f.log
: > "$LOG"
exec >> "$LOG" 2>&1
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
echo "=== window F start $(date +%F' '%H:%M:%S) ==="
echo "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"
echo "--- preflight (expect: nothing) ---"
pgrep -af 'make|nvcc|ptxas|train_dflash2' | grep -v $$ || true
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
free -g | head -2

cd /home/user/ninfer-fusion/build || { echo WINDF_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")

# A .o written by a killed ptxas may be partial while being newer than its source.
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o

rc=1
for attempt in 1 2 3 4; do
  echo "--- make ninfer -j2 attempt $attempt $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  make ninfer -j2 > /tmp/wf_make.log 2>&1
  rc=$?
  tail -6 /tmp/wf_make.log
  [ $rc -eq 0 ] && break
  echo "attempt $attempt failed rc=$rc; retry (make resumes from built .o)"
  sleep 5
done
if [ $rc -ne 0 ]; then
  echo "MAKE_FAILED rc=$rc — 不做验证, 不让陈旧二进制产出绿色结果"
  echo WINDF_FAIL
  exit 3
fi
after=$(stat -c %Y "$BIN")
ls -l --time-style=+%H:%M:%S "$BIN"
if [ "$after" -le "$before" ]; then
  echo "RELINK_MISSING ($before -> $after) — 不验证"
  echo WINDF_FAIL
  exit 3
fi
echo "REBUILT OK ($before -> $after)"

echo "--- [1/3] muse verification $(date +%H:%M:%S) ---"
bash "$J/_muse_verify.sh"
echo "--- [2/3] e8 post-fix probe $(date +%H:%M:%S) ---"
bash "$J/_e8_postfix.sh"
echo "--- [3/3] resume W9 training $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDF_FAIL; exit 3; }
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 30
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "=== window F done $(date +%F' '%H:%M:%S) ==="
echo WINDF_DONE
