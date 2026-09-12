#!/bin/bash
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "############ M_unlanded_now.md"
cat $C/M_unlanded_now.md 2>/dev/null | head -120
echo "############ board.md tail (last 120 lines)"
tail -120 $C/board.md 2>/dev/null
echo "############ a3_scratch listing"
ls -la $C/a3_scratch/ 2>/dev/null
echo "############ a3_scratch/b wrapper (after state)"
cat -n $C/a3_scratch/b/src/ops/wrapper/sigmoid_mul.cpp 2>/dev/null
echo "############ a3_scratch/live wrapper (baseline)"
md5sum $C/a3_scratch/live/src/ops/wrapper/sigmoid_mul.cpp /home/user/ninfer-fusion/src/ops/wrapper/sigmoid_mul.cpp 2>/dev/null
echo "############ a3_scratch scripts"
for f in check_gen.py upd_gen.py counts.py cmp_md5.py; do echo "---- $f"; cat -n $C/a3_scratch/$f 2>/dev/null; done
