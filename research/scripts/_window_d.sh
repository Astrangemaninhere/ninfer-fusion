#!/bin/bash
# Window D: the SIGTERM'd window C died mid force-rebuild, so nothing is linked,
# verified or training. This runs the whole remainder as ONE detached pipeline
# (launched with setsid+nohup so a CLI session interrupt cannot kill it again):
#   rebuild -> ASSERTIVE Muse check -> ASSERTIVE e8 3-tier probe -> resume W9 training
# Hard gates: a failed make or a missing relink aborts BEFORE any verification, so a
# stale binary can never produce a green run.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/window_d.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window D start $(date +%F' '%H:%M:%S) ==="

echo "--- preflight (expect: nothing) ---"
pgrep -af 'make|nvcc|ptxas|train_dflash2' | grep -v $$ || true
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader

cd /home/user/ninfer-fusion/build || { echo WINDD_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")

# ptxas was killed mid-write: a partially written .o can look newer than its source,
# so delete the two decode objects to force a real rebuild instead of a silent link.
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o
echo "--- rebuild $(date +%H:%M:%S) (make ninfer -j2) ---"
make ninfer -j2 > /tmp/wd_make.log 2>&1
rc=$?
tail -8 /tmp/wd_make.log
if [ $rc -ne 0 ]; then
  echo "MAKE_FAILED rc=$rc — 不做验证, 不让陈旧二进制产出绿色结果"
  echo WINDD_FAIL
  exit 3
fi
after=$(stat -c %Y "$BIN")
ls -l --time-style=+%H:%M:%S "$BIN"
if [ "$after" -le "$before" ]; then
  echo "RELINK_MISSING (bin mtime 未更新: $before -> $after) — 不验证"
  echo WINDD_FAIL
  exit 3
fi
echo "REBUILT OK ($before -> $after)"

echo "--- [1/3] muse verification $(date +%H:%M:%S) ---"
bash "$J/_muse_verify.sh"
echo "--- [2/3] e8 post-fix probe $(date +%H:%M:%S) ---"
bash "$J/_e8_postfix.sh"
echo "--- [3/3] resume W9 training $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDD_FAIL; exit 3; }
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 30
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "=== window D done $(date +%F' '%H:%M:%S) ==="
echo WINDD_DONE
