#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "################ OURS dflash2_impl.h 176-215 (draft block prep) ################"
sed -n '176,215p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo
echo "################ speculative_prepare_verify_ids impl ################"
F=$(grep -rl "void speculative_prepare_verify_ids" $R/src/ops 2>/dev/null | head -1)
echo "file: $F"
[ -n "$F" ] && grep -n -B3 -A 45 "void speculative_prepare_verify_ids" "$F"
echo
echo "################ prepare_masked_block impl ################"
F2=$(grep -rl "void prepare_masked_block" $R/src/ops $R/src/targets 2>/dev/null | head -1)
echo "file: $F2"
[ -n "$F2" ] && grep -n -B3 -A 45 "void prepare_masked_block" "$F2"
echo
echo "################ UPSTREAM: verify ids / target block ################"
grep -rn "verify_ids\|prepare_masked_block" $U/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | sed "s|$U/||" | head -25
