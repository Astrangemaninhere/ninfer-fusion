#!/bin/bash
# Window J2 — rebuild after the 256-fix completion (i8 Groups parameterised,
# nvfp4 tiled kernel gated off for non-256 geometries), then TEST THROUGH THE
# SERVER, then resume training LAST.
#
# No patch application here: window J already applied A_s36_headdim.diff to this
# tree and the coordinator applied the two completion edits. Snapshot exists at
# /tmp/pre_j_src.tgz and /tmp/pre_j2_src.tgz for revert.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_j2.log
: > "$LOG"
exec >> "$LOG" 2>&1
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
echo "=== window J2 start $(date +%F' '%H:%M:%S) ==="
free -g | head -2
if pgrep -af train_dflash2 | grep -v $$ > /dev/null; then
  echo "ABORT: training alive"; echo WINDJ2_FAIL; exit 3
fi
tar czf /tmp/pre_j2_src.tgz -C "$R" src 2>/dev/null && echo "snapshot /tmp/pre_j2_src.tgz"

echo "--- [1/3] build (make ninfer -j1) $(date +%H:%M:%S) ---"
cd "$R/build" || { echo WINDJ2_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o
rc=1
for attempt in 1 2 3; do
  echo "--- attempt $attempt $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  make ninfer -j1 > /tmp/wj2_make.log 2>&1
  rc=$?
  tail -4 /tmp/wj2_make.log
  [ $rc -eq 0 ] && break
  echo "--- first errors ---"; grep -E 'error' /tmp/wj2_make.log | head -6 | cut -c1-170
  cp /tmp/wj2_make.log /tmp/wj2_make_attempt$attempt.log
  sleep 5
done
[ $rc -eq 0 ] || { echo "MAKE_FAILED rc=$rc — 不验证"; echo WINDJ2_FAIL; exit 3; }
after=$(stat -c %Y "$BIN")
[ "$after" -gt "$before" ] || { echo "RELINK_MISSING"; echo WINDJ2_FAIL; exit 3; }
echo "REBUILT OK ($before -> $after)"

echo "--- [2/3] judgements THROUGH THE SERVER $(date +%H:%M:%S) ---"
echo "[2a] Muse bf16 + nvfp4 via ninfer-serve (target: zero NaN each; pre-fix 366/239)"
bash "$J/_muse_serve_accept.sh"
echo "[2b] qwen e8 three tiers via ninfer-serve (first REAL e8 data)"
bash "$J/_e8_postfix.sh"
echo "[2c] Muse e8 via ninfer-serve"
bash "$J/_e8_muse_check.sh"

echo "--- [3/3] resume W9 training LAST $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDJ2_FAIL; exit 3; }
powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
sleep 60
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "revert: tar xzf /tmp/pre_j2_src.tgz -C $R"
echo "=== window J2 done $(date +%F' '%H:%M:%S) ==="
echo WINDJ2_DONE
