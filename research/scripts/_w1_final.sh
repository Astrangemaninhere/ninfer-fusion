#!/bin/bash
echo "########## FINAL re-run: generator"
python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/W1_mkpatch.py 2>&1 | tail -12
echo
echo "########## FINAL re-run: verifier"
python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/W1_verify.py 2>&1 | tail -12
echo
echo "########## staged kernel: bank of interest"
grep -n 'QueryTileCausal' /tmp/w1_stage/src/ops/kernel/bidirectional_gqa_attention.cuh
echo
echo "########## staged kernel: gate block (cat -A, first 20 lines)"
sed -n '/Counterfactual causality probe/,/^            score\[nt\]\[0\]/p' /tmp/w1_stage/src/ops/kernel/bidirectional_gqa_attention.cuh | cat -A | sed -e 's/\$$//' | head -22
echo
echo "########## staged swa.cu: knob + lambda"
grep -n 'NINFER_SWA_QUERY_TILE_CAUSAL\|launch_variant\|QueryTileCausal>' /tmp/w1_stage/src/ops/launcher/swa.cu
echo
echo "########## staged dflash2_impl.h: comment (first 10 lines of the insert, CRLF shown as ^M)"
sed -n '253,262p' /tmp/w1_stage/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | cat -A | head -12
echo
echo "########## deliverables:"
ls -la --time-style=long-iso /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/W1_*
