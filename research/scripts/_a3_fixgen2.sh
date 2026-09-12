#!/bin/bash
# A3 headwise fix v2: verify the already-generated patch against the TRUE A3-applied state.
# CPU-only; scratch in /tmp; never writes into the live tree.
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
B=$COL/a3_scratch/b          # A3 generator "after" state (A3-applied, helper missing)
L=$COL/a3_scratch/live       # A3 generator "before" state == live baseline
REL=src/ops/wrapper/sigmoid_mul.cpp
LH=src/ops/launcher/sigmoid_gate_mul.h
FIX=$COL/build/A3_headwise_gate_fix.diff
W=/tmp/a3fix2
rm -rf $W
mkdir -p $W/inc/ops/launcher $W/o1/src/ops/wrapper $W/o2/src/ops/wrapper $W/tree/src/ops/wrapper $W/fixed/src/ops/wrapper

echo "=== 0: confirm the generator states ==="
echo "live/baseline wrapper lines: $(wc -l < $L/$REL) ; after(b) wrapper lines: $(wc -l < $B/$REL)"
grep -c 'headwise_gate_shape' $B/$REL
echo "b/ launcher header lines: $(wc -l < $B/$LH) (live: $(wc -l < $L/$LH))"
cmp $L/$REL $T/$REL && echo "live copy == live tree: IDENTICAL"
cp $B/$LH $W/inc/ops/launcher/sigmoid_gate_mul.h   # A3-applied launcher decl (override)

echo
echo "=== 1: dry-run the fix patch against the TRUE A3-applied wrapper ==="
cp $B/$REL $W/o1/$REL
cd $W/o1 && patch -p1 --dry-run -i $FIX; echo "RC=$?"

echo
echo "=== 2: real apply, order A (A3 first, then fix) ==="
cd $W/o1 && patch -p1 -i $FIX; echo "RC=$?"
echo
echo "=== 3: real apply, order B (fix first on the live baseline, then the A3 headwise diff) ==="
cp -r $L/* $W/o2/ 2>/dev/null
cd $W/o2 && patch -p1 -i $FIX; echo "fix RC=$?"
cd $W/o2 && patch -p1 -i $COL/A3_spark_headwise_gate.diff; echo "A3 RC=$?"
echo "--- cmp order A vs order B wrapper ---"
cmp $W/o1/$REL $W/o2/$REL && echo "ORDER_A == ORDER_B : IDENTICAL"
echo "--- the 6 files after order B (all must be A3-applied) ---"
for f in $REL $LH src/ops/launcher/sigmoid_gate_mul.cu src/ops/kernel/sigmoid_gate_mul.cuh include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp; do
  printf '%-56s %s\n' "$f" "$(md5sum $W/o2/$f | cut -d' ' -f1)"
done

echo
echo "=== 4: host compile checks (g++ -fsyntax-only, exact build flags from compile_commands.json) ==="
CXX=/usr/bin/c++
INC="-I$T/include -I$T/src -I$T/third_party -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include/cccl"
INC_A3="-I$W/inc $INC"   # forces the A3-applied launcher header
echo "--- 4a NEGATIVE CONTROL: A3-applied wrapper, helper missing (expect the reported error) ---"
$CXX -std=gnu++20 -fsyntax-only $INC_A3 $B/$REL; echo "RC=$?"
echo "--- 4b POSITIVE: A3-applied + fix (order A result) ---"
$CXX -std=gnu++20 -fsyntax-only $INC_A3 $W/o1/$REL; echo "RC=$?"
echo "--- 4c POSITIVE: fix alone on the live file, upstream headers only ---"
cp $T/$REL $W/fixed/$REL && cd $W/fixed && patch -p1 -i $FIX >/dev/null && cd $W
$CXX -std=gnu++20 -fsyntax-only $INC $W/fixed/$REL; echo "RC=$?"

echo
echo "=== 5: also dry-run the ORIGINAL A3 diff on the fix-first tree (patch-order safety) ==="
cp -r $L/* $W/tree/ 2>/dev/null
cd $W/tree && patch -p1 -i $FIX >/dev/null 2>&1
cd $W/tree && patch -p1 --dry-run -i $COL/A3_spark_headwise_gate.diff; echo "RC=$?"

echo
echo "=== 6: final state of the patched wrapper (order A, numbered) ==="
cat -n $W/o1/$REL
echo
echo "=== 7: artifacts ==="
ls -la $FIX; md5sum $FIX
echo done
