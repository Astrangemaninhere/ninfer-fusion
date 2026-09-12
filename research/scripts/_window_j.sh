#!/bin/bash
# Window J — land the fixes and TEST THEM THROUGH THE SERVER (user's requirement).
#
# Contents: the 11 patches already applied by window H v2 (n1/n1b/s30/u7/s24/s28/
# s31/s32/s34/s35/s38) + A's unified 256-stride fix (A_s36_headdim.diff, 94 sites)
# which is what should make Muse decode work at all (pre-fix: 366 NaN lines bf16).
# L3 (per-tier TU split) is deliberately EXCLUDED: its apply.sh targeted the wrong
# tree, and memory discipline says keep the original giant TU at -j1.
#
# Judgements all drive `ninfer-serve` over HTTP — a CLI-only pass is not acceptance.
# Training resumes LAST, per the user's ordering.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_j.log
: > "$LOG"
exec >> "$LOG" 2>&1
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
echo "=== window J start $(date +%F' '%H:%M:%S) ==="
free -g | head -2
pgrep -af 'make|nvcc|ptxas|train_dflash2' | grep -v $$ || true
if pgrep -af train_dflash2 | grep -v $$ > /dev/null; then
  echo "ABORT: training alive — pause deliberately first"; echo WINDJ_FAIL; exit 3
fi

echo "--- [1/4] apply the 256-stride fix $(date +%H:%M:%S) ---"
cd "$R" || { echo WINDJ_FAIL; exit 3; }
tar czf /tmp/pre_j_src.tgz -C "$R" src 2>/dev/null && echo "snapshot /tmp/pre_j_src.tgz"
if ! patch -p1 --dry-run < "$J/_collab/A_s36_headdim.diff" > /tmp/j_dryrun.log 2>&1; then
  echo "DRYRUN FAIL"; tail -8 /tmp/j_dryrun.log; echo WINDJ_FAIL; exit 3
fi
grep -cE 'succeeded|checking file' /tmp/j_dryrun.log | sed 's/^/  hunks/files checked: /'
grep -c 'FAILED' /tmp/j_dryrun.log | sed 's/^/  FAILED hunks: /'
if grep -q 'FAILED' /tmp/j_dryrun.log; then echo "hunks failed — abort"; echo WINDJ_FAIL; exit 3; fi
patch -p1 < "$J/_collab/A_s36_headdim.diff" > /tmp/j_apply.log 2>&1 && echo "APPLIED 256-stride fix"
grep -cE '^patching file' /tmp/j_apply.log | sed 's/^/  files patched: /'

echo "--- [2/4] ONE build, -j1 (memory discipline) $(date +%H:%M:%S) ---"
cd "$R/build" || { echo WINDJ_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o \
      src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o
rc=1
for attempt in 1 2 3; do
  echo "--- make ninfer -j1 attempt $attempt $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  make ninfer -j1 > /tmp/wj_make.log 2>&1
  rc=$?
  tail -5 /tmp/wj_make.log
  [ $rc -eq 0 ] && break
  echo "attempt $attempt failed rc=$rc; retry"
  sleep 5
done
[ $rc -eq 0 ] || { echo "MAKE_FAILED rc=$rc — 不验证"; echo WINDJ_FAIL; exit 3; }
after=$(stat -c %Y "$BIN")
[ "$after" -gt "$before" ] || { echo "RELINK_MISSING — 不验证"; echo WINDJ_FAIL; exit 3; }
echo "REBUILT OK ($before -> $after)"

echo "--- [3/4] judgements THROUGH THE SERVER $(date +%H:%M:%S) ---"
echo "[3a] Muse: bf16 + nvfp4 through ninfer-serve (target: zero NaN each)"
bash "$J/_muse_serve_accept.sh"
echo "[3b] qwen: e8 three tiers through ninfer-serve (first REAL e8 data)"
bash "$J/_e8_postfix.sh"
echo "[3c] Muse: e8 through ninfer-serve"
bash "$J/_e8_muse_check.sh"
echo "[3d] CLI cross-check (secondary, debugging aid only)"
bash "$J/_muse_kv_compare.sh"

echo "--- [4/4] resume W9 training LAST $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDJ_FAIL; exit 3; }
powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
sleep 60
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "revert: tar xzf /tmp/pre_j_src.tgz -C $R"
echo "=== window J done $(date +%F' '%H:%M:%S) ==="
echo WINDJ_DONE
