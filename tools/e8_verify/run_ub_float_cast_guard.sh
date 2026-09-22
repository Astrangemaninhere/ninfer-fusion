#!/usr/bin/env bash
# run_ub_float_cast_guard.sh -- build both halves of the guard against the
# `static_cast<int>(rintf(...))` UB class, run the REAL tree (must be green), and run
# SIX INJECTED-DEFECT controls (each must go red). A guard that cannot fail is not
# evidence, so the controls are the load-bearing part.
#
# SAFETY: this script is HOST-ONLY (plain g++, no nvcc, no cmake, no build lock) and it
# NEVER writes inside the tree. Every control is applied to a SHADOW COPY under $OUT.
#
#   UBG_OUT=/some/non-tmpfs/dir bash tools/e8_verify/run_ub_float_cast_guard.sh
#
# $UBG_OUT is REQUIRED and deliberately has no default: the project's rule is that build
# artefacts must not land in a tmpfs, and a silent default is how they do.
set -u
T=/home/user/ninfer-fusion
: "${UBG_OUT:?set UBG_OUT to a non-tmpfs output directory, e.g. UBG_OUT=\$PWD/.ubg}"
OUT="$UBG_OUT"; mkdir -p "$OUT" || exit 1
CT="$OUT/controls"
G="$OUT/ubguard"
RED="$OUT/ubprobe_red"
GRN="$OUT/ubprobe_green"
SAN="-fsanitize=float-cast-overflow -fno-sanitize-recover=float-cast-overflow"
HDR_REL=src/ops/kernel

FIXED=e8_root_codec.cuh
KVQ=gqa_attention_kv_quant.cuh
NVF=gqa_attention_kv_nvfp4.cuh
LAT=e8_lattice.cuh
LATCODE=e8_lattice_codec.cuh

echo "=== ub_float_cast guard + probe ==="
for f in $HDR_REL/$FIXED $HDR_REL/$KVQ $HDR_REL/$NVF $HDR_REL/$LAT $HDR_REL/$LATCODE; do
  echo "  $(sha256sum "$T/$f" | cut -c1-16)  $f"
done
echo

echo "--- building the SOURCE guard (host-only g++) ---"
g++ -O2 -std=c++20 "$T/tools/e8_verify/e8_ub_float_cast_guard.cpp" -o "$G" 2>"$OUT/guard.err"
if [ ! -x "$G" ]; then echo "BUILD FAILED:"; head -30 "$OUT/guard.err"; exit 1; fi
echo "  build OK"

echo "--- building the RUNTIME probe: RED (pre-fix spelling) and GREEN (fixed) ---"
g++ -std=c++20 -O1 -g $SAN -DNINFER_UB_PROBE_RED "$T/tools/e8_verify/ub_float_cast_probe.cpp" -o "$RED" 2>"$OUT/red.err"
g++ -std=c++20 -O1 -g $SAN                           "$T/tools/e8_verify/ub_float_cast_probe.cpp" -o "$GRN" 2>"$OUT/grn.err"
if [ ! -x "$RED" ] || [ ! -x "$GRN" ]; then echo "PROBE BUILD FAILED:"; head -30 "$OUT/red.err" "$OUT/grn.err"; exit 1; fi
echo "  build OK"

show() {
  "$G" "$2" > "$OUT/g_$1.txt" 2>&1
  local rc=$?
  echo "    rc=$rc  $(grep -oE '== [0-9]+/[0-9]+ ==' "$OUT/g_$1.txt" | tail -1)  FAILs=$(grep -c '^FAIL' "$OUT/g_$1.txt")"
  grep -E '^FAIL|SITE |UNREADABLE' "$OUT/g_$1.txt" | head -12 | sed 's/^/      /'
}

mkroot() {
  rm -rf "$CT/$1"; mkdir -p "$CT/$1/$HDR_REL"
  for f in $FIXED $KVQ $NVF $LAT $LATCODE; do cp "$T/$HDR_REL/$f" "$CT/$1/$HDR_REL/$f"; done
}

py() { python3 -c "$@" || { echo "    CONTROL EDIT FAILED"; return 1; }; }

echo
echo "--- REAL TREE (must be rc=0, 0 FAIL) ---"
show real "$T"

echo
echo "--- RUNTIME RED: pre-fix spelling must make the sanitizer ABORT ---"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
"$RED" A > "$OUT/red_A.txt" 2>&1; echo "    RED   (A) rc=$?  $(grep -m1 'runtime error' "$OUT/red_A.txt" | sed 's/^/<< /')"
"$RED" B > "$OUT/red_B.txt" 2>&1; echo "    RED   (B) rc=$?  $(grep -m1 'runtime error' "$OUT/red_B.txt" | sed 's/^/<< /')"
"$GRN" D > "$OUT/grn_D.txt" 2>&1; echo "    GREEN (D) rc=$?  (must be 0, no runtime error)"
"$GRN" C > "$OUT/grn_C.txt" 2>&1; echo "    GREEN (C) rc=$?  $(grep -m1 'max argument' "$OUT/grn_C.txt" | sed 's/^/<< /')"
"$GRN" E > "$OUT/grn_E.txt" 2>&1; echo "    GREEN (E) rc=$?  $(grep -m1 'finite r_rel sweep' "$OUT/grn_E.txt" | sed 's/^/<< /')"

echo
echo "--- CONTROL A: reinstate the UB spelling in e8_root_codec.cuh (must go RED) ---"
mkroot A && py "
p='$CT/A/$HDR_REL/$FIXED'; s=open(p).read()
old='        int q_rad = __float2int_rn(log_val);'
n=s.count(old); print('    anchor count',n); assert n==2,n
open(p,'w').write(s.replace(old,'        int q_rad = static_cast<int>(rintf(log_val));'))
print('    mutant DIFFERS from the real file:', open(p).read()!=open('$T/$HDR_REL/$FIXED').read())
" && show A "$CT/A"

echo
echo "--- CONTROL B: add a NEW UB site in gqa_attention_kv_quant.cuh (must go RED) ---"
mkroot B && py "
p='$CT/B/$HDR_REL/$KVQ'; s=open(p).read()
old='    int q = __float2int_rn(x * inv_scale);'
n=s.count(old); print('    anchor count',n); assert n==1,n
open(p,'w').write(s.replace(old,'    int q = static_cast<int>(rintf(x * inv_scale));'))
print('    a thirteenth writer appeared')
" && show B "$CT/B"

echo
echo "--- CONTROL C: delete the nvfp4 NaN early-return (:147), the premise of unreachability (must go RED) ---"
mkroot C && py "
p='$CT/C/$HDR_REL/$NVF'; s=open(p).read()
m='if (x != x) { return 0; }'
print('    marker count', s.count(m)); assert s.count(m)==1
open(p,'w').write(s.replace(m,'if (false) { return 0; }'))
print('    :160 can now see a NaN')
" && show C "$CT/C"

echo
echo "--- CONTROL D: delete the nvfp4 exponent>=16 saturation (:154) (must go RED) ---"
mkroot D && py "
p='$CT/D/$HDR_REL/$NVF'; s=open(p).read()
m='if (exponent >= 16) {'
print('    marker count', s.count(m), '(two sites: :154 the pre-branch saturation, :175 the in-branch one)'); assert s.count(m)==2
open(p,'w').write(s.replace(m,'if (false) {',1))
print('    the :154 saturation is off, so the guard that keeps :160 free of infinities is gone')
" && show D "$CT/D"

echo
echo "--- CONTROL E: delete a registered header (must not pass vacuously, must go RED) ---"
mkroot E && rm -f "$CT/E/$HDR_REL/$LAT" && show E "$CT/E"

echo
echo "--- CONTROL F: delete the FIX's documented reason from the site (must go RED) ---"
mkroot F && py "
p='$CT/F/$HDR_REL/$FIXED'; s=open(p).read()
m='__float2int_rn, NOT static_cast<int>(rintf(log_val))'
print('    marker count', s.count(m)); assert s.count(m)==2
open(p,'w').write(s.replace(m,'the conversion'))
print('    the reason is no longer documented')
" && show F "$CT/F"

echo
echo "=== real-tree guard output (full) ==="
cat "$OUT/g_real.txt"
echo
echo "=== real-tree probe output (full) ==="
cat "$OUT/grn_D.txt"; cat "$OUT/grn_C.txt"
