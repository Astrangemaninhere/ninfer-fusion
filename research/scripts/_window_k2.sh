#!/bin/bash
# Window K2 — GPU ordering that respects "one model at a time", replacing window K's
# plan (which would have rebuilt decode.cu for a stride patch that is bit-identical at
# head_dim 256, then grabbed the GPU with its own judgements before the Patch A
# measurement could run).
#
#   wait J2 -> ensure a patched binary exists -> wait for the Patch A measurement
#   (exactness + acceptance, run by _post_build_measure.sh) -> spec 4-way -> Muse
#   -> resume W9 training LAST
#
# Deliberately NOT done here (documented in _TODO.md): E3's i8 plane-stride patch and
# the forced decode-TU rebuild — both are no-ops at head_dim 256 (bit-identical
# offsets) and the decode TU costs ~65 min under -j1/22 GB. Land them with the Muse
# work instead.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_k2.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window K2 start $(date +%F' '%H:%M:%S) ==="
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"

echo "--- [1/6] wait for window J2 $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -qE 'WINDJ2_(DONE|FAIL)' "$J/dl/window_j2.log" 2>/dev/null; do
  sleep 20
  if [ $(( $(date +%s) - t0 )) -gt 14400 ]; then echo "TIMEOUT waiting for J2"; break; fi
done
tail -5 "$J/dl/window_j2.log" | cut -c1-150
free -g | head -2

echo "--- [2/6] binary must be newer than Patch A $(date +%H:%M:%S) ---"
BIN=$R/build/apps/ninfer-serve
PA=$(stat -c %Y $R/src/targets/qwen3_6/impl/runtime/program_impl.h)
for i in 1 2 3 4; do
  bm=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
  if [ "$bm" -gt "$PA" ]; then echo "  binary $(date -d @$bm '+%m-%d %H:%M') is newer than Patch A $(date -d @$PA '+%m-%d %H:%M')"; break; fi
  echo "  attempt $i: rebuilding (binary $(date -d @$bm '+%m-%d %H:%M'))"
  free -g | sed -n 2p
  ( cd $R/build && make ninfer -j1 > /tmp/wk2_make_$i.log 2>&1 )
  rc=$?
  tail -4 /tmp/wk2_make_$i.log
  if [ $rc -ne 0 ]; then grep -E 'error' /tmp/wk2_make_$i.log | head -6 | cut -c1-170; else echo "  make ok"; fi
  sleep 5
done
bm=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
[ "$bm" -gt "$PA" ] && echo "  OK patched binary in place" || { echo "NO_PATCHED_BINARY"; echo WINDK2_FAIL; exit 3; }

echo "--- [3/6] wait for the Patch A measurement to finish $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -qE 'POSTBUILD_(DONE|TIMEOUT)' "$J/dl/postbuild_measure.log" 2>/dev/null; do
  sleep 20
  if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then echo "TIMEOUT waiting for the Patch A measure"; break; fi
done
grep -E 'IDENTICAL|DIFFERENT|SERVE_FAILED|^\| ' "$J/dl/postbuild_measure.log" | tail -14
echo "  Patch A verdict file:"; cat /home/user/pb_verdict.txt 2>/dev/null | head -20

echo "--- [4/6] four-way spec comparison $(date +%H:%M:%S) ---"
bash "$J/_spec_4way.sh" 2>&1 | tail -30

echo "--- [5/6] Muse acceptance via serve $(date +%H:%M:%S) ---"
bash "$J/_muse_serve_accept.sh" 2>&1 | tail -30

echo "--- [6/6] resume W9 training LAST $(date +%H:%M:%S) ---"
ls -l "$J/_train_df2_resume.bat" 2>/dev/null | cut -c1-100 || echo "  (resume bat absent)"
if [ -f "$J/_train_df2_resume.bat" ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
  sleep 60
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
fi
echo "=== window K2 done $(date +%F' '%H:%M:%S) ==="
echo WINDK2_DONE
