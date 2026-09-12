#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "################ who writes the acceptance decision ################"
grep -rn "accepted_drafts\|licensed_counts\|target_argmax" $R/src $R/include 2>/dev/null \
  | sed "s|$R/||" | grep -v "\.orig:" | head -40
echo
echo "################ acceptance kernels by filename ################"
grep -rln "accepted_drafts\|licensed_counts" $R/src/ops $R/src/targets 2>/dev/null | sed "s|$R/||"
echo
echo "################ run progress ################"
tail -12 $J/dl/df2_align.log 2>/dev/null
echo "--- probe raw so far ---"
wc -l $J/dl/df2_probe_raw.log 2>/dev/null
grep -c "df2dbg" $J/dl/df2_probe_raw.log 2>/dev/null
