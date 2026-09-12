#!/bin/bash
# Window K — second landing pass. Waits for window J2 to finish, then:
#   1. forces the decode TU to be rebuilt (A's i8 stride patch landed *while*
#      that TU was compiling, so its .o will look newer than the source — the
#      silent-stale trap),
#   2. applies whatever new patches exist (acceptance counter, i8 plane stride,
#      cold-policy F3b fix, dflash2 verify fix) as an optional tier,
#   3. fixes the `--spec` usage-text gap (parser accepts mtp|dflash|dflash2|auto,
#      usage lists only two — the exact gap that cost two wrong measurement runs),
#   4. ONE build, then the serve-side judgements: the four-way spec comparison and
#      the Muse acceptance,
#   5. resumes W9 training LAST.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_k.log
: > "$LOG"
exec >> "$LOG" 2>&1
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
echo "=== window K start $(date +%F' '%H:%M:%S) ==="

echo "--- [0/5] wait for window J2 to be completely done $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -qE 'WINDJ2_(DONE|FAIL)' "$J/dl/window_j2.log" 2>/dev/null; do
  sleep 20
  if [ $(( $(date +%s) - t0 )) -gt 10800 ]; then echo "TIMEOUT waiting for J2"; break; fi
done
tail -4 "$J/dl/window_j2.log" | cut -c1-140
free -g | head -2

echo "--- [1/5] force the stride-affected TUs to rebuild $(date +%H:%M:%S) ---"
cd "$R" || { echo WINDK_FAIL; exit 3; }
tar czf /tmp/pre_k_src.tgz -C "$R" src 2>/dev/null && echo "snapshot /tmp/pre_k_src.tgz"
touch src/ops/kernel/gqa_attention_decode_i8.cuh src/ops/kernel/cold_i8_kernels.cuh
echo "touched decode_i8.cuh + cold_i8_kernels.cuh"

echo "--- [2/5] apply the new patches (optional tier) $(date +%H:%M:%S) ---"
declare -A PATCHES=(
  [acceptance_counter]="$J/_collab/E2_s44_acceptance_counter.diff"
  [i8_plane_stride]="$J/_collab/E3_s45_i8_plane_stride.diff"
  [cold_policy_fix]="$J/_collab/E4_s46_cold_policy_fix.diff"
  [df2_verify]="$J/_collab/E_s43_df2_verify.diff"
)
for name in "${!PATCHES[@]}"; do
  p="${PATCHES[$name]}"
  if [ ! -e "$p" ]; then echo "  skip (absent) $name"; continue; fi
  if patch -p1 --dry-run < "$p" > /tmp/k_$name.log 2>&1; then
    patch -p1 < "$p" >> /tmp/k_$name.log 2>&1 && echo "  applied $name"
  else
    echo "  DRYRUN FAIL $name"; tail -3 /tmp/k_$name.log
  fi
done

echo "--- [3/5] fix the --spec usage text $(date +%H:%M:%S) ---"
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("/home/user/ninfer-fusion/src/serve/serve_options.cpp")
t = p.read_text(encoding="utf-8", errors="surrogateescape")
old = "--spec mtp|dflash"
new = "--spec mtp|dflash|dflash2|auto"
if old in t and new not in t:
    p.write_text(t.replace(old, new), encoding="utf-8", errors="surrogateescape")
    print("  usage text updated: %s -> %s" % (old, new))
else:
    print("  usage text: nothing to do (already fixed or pattern absent)")
PY

echo "--- [4/5] ONE build $(date +%H:%M:%S) ---"
cd "$R/build" || { echo WINDK_FAIL; exit 3; }
BIN=apps/ninfer
before=$(stat -c %Y "$BIN")
rm -f src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o
rc=1
for attempt in 1 2 3; do
  echo "--- attempt $attempt $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  make ninfer -j1 > /tmp/wk_make.log 2>&1
  rc=$?
  tail -4 /tmp/wk_make.log
  [ $rc -eq 0 ] && break
  grep -E 'error' /tmp/wk_make.log | head -5 | cut -c1-170
  cp /tmp/wk_make.log /tmp/wk_make_attempt$attempt.log
  sleep 5
done
[ $rc -eq 0 ] || { echo "MAKE_FAILED rc=$rc — 不验证"; echo WINDK_FAIL; exit 3; }
after=$(stat -c %Y "$BIN")
[ "$after" -gt "$before" ] || { echo "RELINK_MISSING"; echo WINDK_FAIL; exit 3; }
echo "REBUILT OK ($before -> $after)"

echo "--- [5/5] judgements through the server $(date +%H:%M:%S) ---"
echo "[5a] four-way spec comparison (baseline / mtp / dflash / dflash2)"
bash "$J/_spec_4way.sh"
echo "[5b] Muse acceptance via serve"
bash "$J/_muse_serve_accept.sh"
echo "[5c] dflash2 acceptance counter check (if the counter patch landed)"
grep -hoE 'spec_(drafted|accepted|accept_rate|rounds)=[0-9.]+' /home/user/s4w_dflash2.log 2>/dev/null | sort -u | head -6

echo "--- resume W9 training LAST $(date +%H:%M:%S) ---"
cd "$J" || { echo WINDK_FAIL; exit 3; }
powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
sleep 60
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo "revert: tar xzf /tmp/pre_k_src.tgz -C $R"
echo "=== window K done $(date +%F' '%H:%M:%S) ==="
echo WINDK_DONE
