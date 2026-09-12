#!/bin/bash
echo "=== dflash2 + lm-head-draft error ==="
grep -nE 'error|Error|throw|abort|invalid|available|terminate' /home/user/lmh2_df2_lmhd.log | head -12
echo "--- last 12 lines ---"
tail -12 /home/user/lmh2_df2_lmhd.log
echo
echo "=== did dspark really use the optimized head? ==="
echo "proposal_head mentions: $(grep -c 'proposal_head' /home/user/lmh_dspark_lmhd.log)"
grep -oE '"proposal_head"[^,}]*' /home/user/lmh_dspark_lmhd.log | head -3
grep -oE 'proposal_head[^,}]{0,30}' /home/user/lmh_dspark_lmhd.log | head -5
echo
echo "=== dflash2 full-head request-log head field ==="
grep -oE 'proposal_head[^,}]{0,30}' /home/user/lmh2_df2_full.log | head -3
