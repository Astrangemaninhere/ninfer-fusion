#!/bin/bash
# Window E: window C and D were both killed (C by a session SIGTERM, D by the
# kernel OOM killer at 11:09/11:10 — two parallel ptxas peaked at ~23GB against
# this WSL's 24GB cap, journalctl shows global_oom / Killed process ptxas).
# So: serialise the compiles (-j1 => one ptxas ~14GB) and RETRY the make, since
# make resumes from the .o files that already landed. Then the same gates:
# relink proof -> ASSERTIVE Muse check -> ASSERTIVE e8 3-tier probe -> resume W9.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/window_e.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window E start $(date +%F' '%H:%M:%S) ==="
echo "--- preflight (expect: nothing) ---"
pgrep -af 'make|nvcc|ptxas|train_dflash2' | grep -v $$ || true
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
free -g | head -2

cd /home/user/ninfer-fusion/build || { echo WINDE_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")

# Both decode .o were being written when ptxas was killed; a partial .o is newer
# than its source and would be linked silently. Delete them so they rebuild.
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o

rc=1
for attempt in 1 2 3; do
  echo "--- make ninfer -j1 attempt $attempt $(date +%H:%M:%S) ---"
  make ninfer -j1 > /tmp/we_make.log 2>&1
  rc=$?
  tail -6 /tmp/we_make.log
  [ $rc -eq 0 ] && break
  echo "attempt $attempt failed rc=$rc; retrying (make resumes from built .o)"
  sleep 5
done
if [ $rc -ne 0 ]; then
  echo "MAKE_FAILED rc=$rc after 3 attempts — 不做验证, 不让陈旧二进制产出绿色结果"
  echo WINDE_FAIL
  exit 3
fi
after=$(stat -c %Y "$BIN")
ls -l --time-style=+%H:%M:%S "$BIN"
if [ "$after" -le "$before" ]; then
  echo "RELINK_MISSING (bin mtime 未更新: $before -> $after) — 不验证"
  echo WINDE_FAIL
  exit 3
fi
echo "REBUILT OK ($before -> $after)"

echo "--- [1/3] muse verification $(date +%H:%M:%S) ---"
bash "$J/_muse_verify.sh"
echo "--- [2/3] e8 post-fix probe $(date +%H:%M:%S) ---"
bash "$J/_e8_postfix.sh"
echo "--- [3/3] resume W9 training $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDE_FAIL; exit 3; }
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 30
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "=== window E done $(date +%F' '%H:%M:%S) ==="
echo WINDE_DONE
