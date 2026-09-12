#!/bin/bash
# Patch A build orchestrator.
#
# Facts this is built on (measured, not assumed):
#  * make's recorded prerequisites for ninfer_engine's variant.cpp.o are only
#    {compiler_depend.ts, flags.make, the .cpp} -> editing program_impl.h does NOT
#    schedule a rebuild (no compiler_depend.make in this tree). So Patch A needs the
#    *source* touched.
#  * ninfer_ops' gqa_attention_decode.cu.o (15:33) is up to date by make's own
#    account, so window J2's in-flight rebuild of it (65+ min) is redundant work.
#  * the variant objects are tiny (2.1-2.2 MB) -> the rebuild is cheap.
#
# Steps: kill the redundant build -> touch the variant sources -> one build ->
# only then release window K2 (its wait key is WINDJ2_(DONE|FAIL) in window_j2.log).
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/patchA_build.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== Patch A build $(date '+%F %H:%M:%S') ==="

echo "--- [0] dependency-tracking evidence ---"
ls -l $R/build/src/CMakeFiles/ninfer_engine.dir/compiler_depend.make 2>&1 | cut -c1-110
grep -c 'program_impl\.h' $R/build/src/CMakeFiles/ninfer_engine.dir/compiler_depend.make 2>/dev/null || echo "  (no compiler_depend.make)"

echo "--- [1] stop window J2's redundant build ---"
for p in $(pgrep -f '_window_j2.sh' 2>/dev/null); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in *_window_j2.sh*) echo "  TERM harness $p"; kill -TERM "$p" 2>/dev/null;; esac
done
sleep 2
for p in $(pgrep -x make) $(pgrep -x nvcc) $(pgrep -x ptxas) $(pgrep -x cicc); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *ninfer-fusion*|*nvcc*|*ptxas*|*cicc*|*make*) echo "  TERM $p $(echo "$cmd" | cut -c1-50)"; kill -TERM "$p" 2>/dev/null;;
  esac
done
sleep 5
echo "  survivors: $(pgrep -c -x nvcc 2>/dev/null || echo 0) nvcc, $(pgrep -c -x make 2>/dev/null || echo 0) make"
free -g | head -2
echo "  decode TU object (must survive): $(ls -l --time-style=+%H:%M $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o 2>/dev/null | awk '{print $6, $5}')"

echo "--- [2] force the Patch A TUs (touch sources, not headers) ---"
for f in $R/src/targets/qwen3_6_27b/impl/variant.cpp \
         $R/src/targets/muse_glimmer_30b/impl/variant.cpp \
         $R/src/targets/qwen3_6_35b_a3b/impl/variant.cpp; do
  [ -f "$f" ] && touch "$f" && echo "  touched $(basename $(dirname $(dirname $f)))/$(basename $f)"
done

echo "--- [3] build plan (what make now intends to do) ---"
cd $R/build || exit 3
timeout 180 make -n ninfer 2>&1 | grep -oE 'Building (CUDA|CXX) object [^ ]*|Linking [^ ]*libninfer[a-z_]*\.a|Linking CXX executable ninfer' | sed 's/.*Building /  build /; s/.*Linking /  link /' | sort -u | head -20

echo "--- [4] build $(date +%H:%M:%S) ---"
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
rc=1
for attempt in 1 2 3 4; do
  echo "  attempt $attempt $(date +%H:%M:%S)"
  free -g | sed -n 2p
  make ninfer -j1 > /tmp/pa_make_$attempt.log 2>&1
  rc=$?
  tail -3 /tmp/pa_make_$attempt.log
  [ $rc -eq 0 ] && break
  grep -E 'error|Error' /tmp/pa_make_$attempt.log | head -6 | cut -c1-160
  sleep 5
done
BIN=$R/build/apps/ninfer-serve
if [ $rc -ne 0 ]; then
  echo "MAKE_FAILED rc=$rc"
else
  echo "MAKE OK $(date +%H:%M:%S)"
fi
echo "  binaries:"
ls -l --time-style=+%m-%d_%H:%M $R/build/apps/ninfer $R/build/apps/ninfer-serve 2>/dev/null | awk '{print "   ", $6, $5, $NF}'
echo "  fresh variant objects:"
ls -l --time-style=+%m-%d_%H:%M $R/build/src/CMakeFiles/ninfer_engine.dir/targets/*/impl/variant.cpp.o 2>/dev/null | awk '{printf "    %s %s %s\n", $6, $5, $NF}'

echo "--- [5] release window K2 (write its wait key) ---"
{
  echo "--- killed by M: redundant decode rebuild; Patch A built by _patchA_build.sh ---"
  if [ $rc -eq 0 ]; then echo "WINDJ2_DONE"; else echo "WINDJ2_FAIL"; fi
} >> "$J/dl/window_j2.log"
echo "=== patchA build done $(date '+%F %H:%M:%S') rc=$rc ==="
