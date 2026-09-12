#!/bin/bash
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
FIX=$COL/build/A3_headwise_gate_fix.diff
EV=$COL/build/A3_headwise_fix_evidence.txt

{
echo
echo "==========================================================================="
echo "[E10] acceptance run straight against the LIVE tree (dry-run only, no writes)"
echo "==========================================================================="
cd $T
echo "\$ cd $T && patch -p1 --dry-run < $FIX"
patch -p1 --dry-run < $FIX; echo "rc=$?"
echo "--- live file md5 after the dry-run (must still equal the baseline) ---"
md5sum $T/src/ops/wrapper/sigmoid_mul.cpp
echo "--- reverse dry-run on the live tree (rc=1 expected: nothing to undo yet) ---"
patch -p1 -R --dry-run < $FIX; echo "rc=$?"
} >> $EV 2>&1

echo "--- cleanup of throwaway scratch files (keep the two harness scripts) ---"
rm -f /mnt/c/Users/User/Documents/ziqinzhang/_tmp_patch_view.sh \
      /mnt/c/Users/User/Documents/ziqinzhang/_ev_tail.txt \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe1.sh /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe1.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe2.sh /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe2.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe3.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe4.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe5.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe6.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe7.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe8.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_probe9.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen2.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen3.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen4.out \
      /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen5.out /mnt/c/Users/User/Documents/ziqinzhang/_a3_fixgen6.out
echo "kept: _a3_fixgen6.sh (evidence generator), _a3_revertcheck.sh (revert proof), _a3_report_draft.md (report source)"
ls -la /mnt/c/Users/User/Documents/ziqinzhang/_a3_* 2>/dev/null
echo
echo "--- final deliverables ---"
ls -la $COL/build/A3_headwise_gate_fix.diff $COL/build/A3_headwise_fix_report.md $COL/build/A3_headwise_fix_evidence.txt
md5sum $COL/build/A3_headwise_gate_fix.diff
echo "--- live tree unchanged (all 6 headwise files) ---"
cd $T && md5sum src/ops/wrapper/sigmoid_mul.cpp src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu src/ops/kernel/sigmoid_gate_mul.cuh include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp
