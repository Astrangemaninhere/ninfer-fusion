#!/usr/bin/env bash
# run_dead_guard.sh -- build tools/e8_verify/e8_dead_guard_test.cpp, run it against the REAL
# tree (must be green), then against FIVE INJECTED-DEFECT copies (each must go red).
# A guard that cannot fail is not evidence, so the controls are the load-bearing part.
set -u
T=/home/user/ninfer-fusion
G=/tmp/e8dg
CT=/tmp/e8dg_controls
CPP_REL=src/targets/qwen3_6/impl/state/decoder_state.cpp
HDR_REL=src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h

echo "=== e8_dead_guard_test ==="
echo "test sha16: $(sha256sum "$T/tools/e8_verify/e8_dead_guard_test.cpp" | cut -c1-16)"
echo "cpp sha16 : $(sha256sum "$T/$CPP_REL" | cut -c1-16)"
echo "hdr sha16 : $(sha256sum "$T/$HDR_REL" | cut -c1-16)"
echo
g++ -O2 -std=c++17 -I"$T/src" "$T/tools/e8_verify/e8_dead_guard_test.cpp" -o "$G" 2>/tmp/e8dg.err
if [ ! -x "$G" ]; then echo "BUILD FAILED:"; head -25 /tmp/e8dg.err; exit 1; fi
echo "build: OK"

mkroot() {
  rm -rf "$CT/$1"; mkdir -p "$CT/$1/$(dirname $CPP_REL)" "$CT/$1/$(dirname $HDR_REL)"
  cp "$T/$CPP_REL" "$CT/$1/$CPP_REL"; cp "$T/$HDR_REL" "$CT/$1/$HDR_REL"
}

show() {
  "$G" "$2" > "/tmp/e8dg_$1.txt" 2>&1
  local rc=$?
  echo "    rc=$rc  $(grep -oE '== [0-9]+/[0-9]+ ==' "/tmp/e8dg_$1.txt" | tail -1)  FAILs=$(grep -c '^FAIL' "/tmp/e8dg_$1.txt")"
  grep -E '^FAIL|OFFENDER' "/tmp/e8dg_$1.txt" | sed 's/^/      /'
}

echo
echo "--- REAL TREE (must be rc=0, 0 FAIL) ---"
show real "$T"

echo
echo "--- CONTROL A: reinstate the dead ternary on BOTH accessors (CODE, must go RED) ---"
mkroot A
python3 -c "
p='$CT/A/$CPP_REL'; s=open(p).read()
old='    const DType layer_dtype = layer_dtypes_[layer];'
new='    const DType layer_dtype = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer];'
n=s.count(old); print('    anchor count',n); assert n==2,n
open(p,'w').write(s.replace(old,new))
print('    mutant DIFFERS from the real file:', open(p).read()!=open('$T/$CPP_REL').read())
"
show A "$CT/A"

echo
echo "--- CONTROL B: change the table declaration's TYPE (premise broken, must go RED) ---"
mkroot B
python3 -c "
p='$CT/B/$HDR_REL'; s=open(p).read()
old='std::array<DType, 64> layer_dtypes{};'
n=s.count(old); print('    anchor count',n); assert n==1,n
open(p,'w').write(s.replace(old,'std::vector<DType> layer_dtypes{};'))
print('    layer_dtypes is now a std::vector, whose empty() WOULD be real')
"
show B "$CT/B"

echo
echo "--- CONTROL C: delete the header (must not pass vacuously, must go RED) ---"
mkroot C
rm -f "$CT/C/$HDR_REL"
show C "$CT/C"

echo
echo "--- CONTROL D: delete the documented reason from the header (must go RED) ---"
mkroot D
python3 -c "
p='$CT/D/$HDR_REL'; s=open(p).read()
m='NO \"table absent\" state'
print('    marker count', s.count(m)); assert s.count(m)==1
open(p,'w').write(s.replace(m,'no such state'))
print('    the reason is no longer documented')
"
show D "$CT/D"

echo
echo "--- CONTROL E: restore the stale MTP comment, which read .empty() as a signal (must go RED) ---"
mkroot E
python3 -c "
p='$CT/E/$CPP_REL'; s=open(p).read()
anchor='        // authoritative for them too. (This used to read \"...through'
print('    anchor count', s.count(anchor)); assert s.count(anchor)==1
s=s.replace(anchor, '        // they follow the same V codec rule through layer_dtypes_.empty().\n        // x (This used to read \"...through')
open(p,'w').write(s)
print('    stale comment restored')
"
show E "$CT/E"

echo
echo "=== real-tree guard output (full) ==="
cat /tmp/e8dg_real.txt
