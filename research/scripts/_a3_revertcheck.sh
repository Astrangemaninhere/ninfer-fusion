#!/bin/bash
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
B=$COL/a3_scratch/b
L=$COL/a3_scratch/live
REL=src/ops/wrapper/sigmoid_mul.cpp
FIX=$COL/build/A3_headwise_gate_fix.diff
W=/tmp/a3rev
rm -rf $W; mkdir -p $W/a/src/ops/wrapper $W/b/src/ops/wrapper
cp $L/$REL $W/a/$REL; (cd $W/a && patch -p1 -i $FIX >/dev/null)
cp $B/$REL $W/b/$REL; (cd $W/b && patch -p1 -i $FIX >/dev/null)

echo "=== reverse dry-run on the fix-alone state (expect rc=0, back to baseline) ==="
cd $W/a && patch -p1 -R --dry-run -i $FIX; echo "rc=$?"
echo "=== reverse dry-run on the A3+fix state (expect rc=0; leaves the A3-only/broken state) ==="
cd $W/b && patch -p1 -R --dry-run -i $FIX; echo "rc=$?"
echo "=== real reverse on the fix-alone state: must restore the baseline bytes ==="
cd $W/a && patch -p1 -R -i $FIX >/dev/null && md5sum $W/a/$REL $L/$REL
cmp $W/a/$REL $L/$REL && echo "RESTORED == baseline: IDENTICAL"
echo "=== full revert of BOTH patches (order: fix then A3) on a scratch copy ==="
mkdir -p $W/full && cp -r $L/* $W/full/ 2>/dev/null
cd $W/full && patch -p1 -i $FIX >/dev/null && patch -p1 -i $COL/A3_spark_headwise_gate.diff >/dev/null
cd $W/full && patch -p1 -R -i $COL/A3_spark_headwise_gate.diff >/dev/null && patch -p1 -R -i $FIX >/dev/null
for f in $REL src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu src/ops/kernel/sigmoid_gate_mul.cuh include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp; do
  cmp -s $W/full/$f $L/$f && echo "IDENTICAL to baseline: $f" || echo "DIFFERS: $f"
done
echo "=== expected md5 values for the report ==="
md5sum $FIX
