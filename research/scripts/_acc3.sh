#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ speculative_round.cuh : accept logic ################"
grep -n "accepted\|argmax\|drafts\[" $R/src/ops/kernel/speculative_round.cuh | head -60
echo
echo "################ speculative_prepare_verify_ids ################"
grep -rn "speculative_prepare_verify_ids" $R/src/ops $R/include 2>/dev/null | sed "s|$R/||"
echo "--- impl ---"
F=$(grep -rl "void speculative_prepare_verify_ids" $R/src/ops 2>/dev/null | head -1)
echo "file: $F"
[ -n "$F" ] && grep -n -A 40 "void speculative_prepare_verify_ids" "$F" | head -60
