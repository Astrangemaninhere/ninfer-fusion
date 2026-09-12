#!/bin/bash
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "############ launcher header"
cat -n $T/src/ops/launcher/sigmoid_gate_mul.h
echo "############ launcher cu"
cat -n $T/src/ops/launcher/sigmoid_gate_mul.cu
echo "############ kernel cuh"
cat -n $T/src/ops/kernel/sigmoid_gate_mul.cuh
echo "############ test cpp"
cat -n $T/tests/ops/test_sigmoid_mul.cpp
echo "############ op_tester.h location"
find $T -name 'op_tester.h' -not -path '*/build/*' 2>/dev/null
echo "############ _batch_build_reconfig.sh"
ls -la /home/user/_batch_build_reconfig.sh 2>/dev/null || echo "not in /home/user"
find /home/user -maxdepth 1 -name '_batch*' 2>/dev/null
echo "############ live tree mtime now"
stat -c '%y %n' $T/src/ops/wrapper/sigmoid_mul.cpp $T/tests/ops/test_sigmoid_mul.cpp $T/src/ops/kernel/sigmoid_gate_mul.cuh 2>/dev/null
echo "############ live_md5_after vs current, all 6 files"
cd $T && md5sum src/ops/wrapper/sigmoid_mul.cpp src/ops/kernel/sigmoid_gate_mul.cuh src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp
echo "---- live_md5.txt (A3 baseline snapshot)"
grep -E 'sigmoid_mul|sigmoid_gate_mul|test_sigmoid' $C/a3_scratch/live_md5.txt
echo "---- live_md5_after.txt"
grep -E 'sigmoid_mul|sigmoid_gate_mul|test_sigmoid' $C/a3_scratch/live_md5_after.txt
echo "############ shadows/headwise_gate"
find $C/a3_scratch/shadows -type f 2>/dev/null | head -20
echo "############ dryrun_headwise_gate.txt"
cat $C/a3_scratch/dryrun_headwise_gate.txt
