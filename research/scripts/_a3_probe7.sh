#!/bin/bash
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "############ grep -n -C3 headwise in collab md docs"
for f in "$C/A3_spark_newops.md" "$C/board.md" "$C/M_unlanded_now.md" "$C/build/S2_generator_interface.md" "$C/build/T2_tree_state_ledger.md"; do
  echo "==== $f"
  grep -n -i -C 4 'headwise' "$f" 2>/dev/null | head -80
done
echo "############ _collab/build stage dirs"
for d in staged staged2 backup evidence s3_scratch; do echo "---- $d"; ls $C/build/$d 2>/dev/null | head -25; done
echo "############ grep headwise_gate_shape in collab (context)"
grep -rn -C2 'headwise_gate_shape' $C 2>/dev/null | grep -v a3_scratch/b | head -40
echo "############ op_tester.h"
cat -n $T/tests/ops/op_tester.h 2>/dev/null | head -200
echo "############ core/tensor.h (head)"
cat -n $T/include/ninfer/core/tensor.h 2>/dev/null | head -120
echo "############ tensor.cpp is_contiguous"
grep -n -A20 'is_contiguous' $T/src/core/tensor.cpp 2>/dev/null | head -60
