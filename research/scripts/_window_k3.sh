#!/bin/bash
# Window K3 — the GPU lane, with the serve-target gap fixed.
#
# Why K3 replaces K2: `make ninfer` does NOT build apps/ninfer-serve, so the binary
# every measurement uses would have stayed at 07:32 (pre-Patch-A) and both K2's check
# and the post-build watcher would have waited forever. K3 links ninfer-serve
# explicitly, then hands the GPU over in priority order.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_k3.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window K3 start $(date +%F' '%H:%M:%S) ==="
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"

echo "--- [1/6] wait for the Patch A build $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -q 'patchA build done' "$J/dl/patchA_build.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { echo "TIMEOUT waiting for patchA build"; break; }
done
tail -6 "$J/dl/patchA_build.log" | cut -c1-140

echo "--- [2/6] link ninfer-serve (Patch A) $(date +%H:%M:%S) ---"
BIN=$R/build/apps/ninfer-serve
PA=$(stat -c %Y $R/src/targets/qwen3_6/impl/runtime/program_impl.h)
cd $R/build || exit 3
for i in 1 2 3; do
  bm=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
  if [ "$bm" -gt "$PA" ]; then echo "  serve already newer than Patch A"; break; fi
  echo "  attempt $i: make ninfer-serve -j1"
  free -g | sed -n 2p
  make ninfer-serve -j1 > /tmp/k3_serve_$i.log 2>&1
  rc=$?
  tail -3 /tmp/k3_serve_$i.log
  [ $rc -ne 0 ] && grep -E 'error|Error' /tmp/k3_serve_$i.log | head -5 | cut -c1-160
  sleep 5
done
bm=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
echo "  ninfer-serve: $(date -d @$bm '+%m-%d %H:%M:%S') size=$(stat -c %s "$BIN" 2>/dev/null)"
if [ "$bm" -gt "$PA" ]; then echo "  PATCHED_SERVE_OK"; else echo "  NO_PATCHED_SERVE — measurements would be meaningless"; fi

echo "--- [3/6] wait for the Patch A measurement $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -qE 'POSTBUILD_(DONE|TIMEOUT)' "$J/dl/postbuild_measure.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "TIMEOUT waiting for the measurement"; break; }
done
grep -E 'IDENTICAL|DIFFERENT|SERVE_FAILED|^\| ' "$J/dl/postbuild_measure.log" 2>/dev/null | tail -16
echo "  --- verdict ---"
cat /home/user/pb_verdict.txt 2>/dev/null | head -24

echo "--- [4/6] four-way spec comparison $(date +%H:%M:%S) ---"
bash "$J/_spec_4way.sh" 2>&1 | tail -40

echo "--- [5/6] Muse acceptance via serve $(date +%H:%M:%S) ---"
bash "$J/_muse_serve_accept.sh" 2>&1 | tail -30

echo "--- [6/6] resume W9 training LAST $(date +%H:%M:%S) ---"
if [ -f "$J/_train_df2_resume.bat" ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
  sleep 60
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
else
  echo "  (resume bat absent — training NOT restarted)"
fi
echo "=== window K3 done $(date +%F' '%H:%M:%S) ==="
echo WINDK3_DONE
