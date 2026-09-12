#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ prepare_masked_block kernel ################"
F=$(ls $R/src/ops/kernel/prepare_masked_block* 2>/dev/null | head -1)
echo "file: $F"
[ -n "$F" ] && cat "$F"
echo
echo "################ launcher ################"
F2=$(grep -rl "prepare_masked_block_launch" $R/src/ops/launcher 2>/dev/null | head -1)
echo "file: $F2"
[ -n "$F2" ] && cat "$F2"
