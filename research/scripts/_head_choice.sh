#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "################ UPSTREAM serve_options around the Optimized choice ################"
sed -n '270,300p' $U/src/serve/serve_options.cpp
echo
echo "################ OURS serve_options: any proposal_head choice? ################"
grep -rn "proposal_head" $R/src/serve $R/apps 2>/dev/null | sed "s|$R/||" | head -20
echo "--- (context) ---"
grep -rn -B6 -A6 "proposal_head" $R/src/serve/serve_options.cpp 2>/dev/null | head -40
echo
echo "################ UPSTREAM: how proposal_head reaches the plan ################"
grep -rn "proposal_head" $U/src/targets/qwen3_6/impl/runtime/program_impl.h $U/src/targets/qwen3_6/impl/runtime/layouts_impl.h 2>/dev/null | sed "s|$U/||" | head -20
echo
echo "################ OURS: does ops::linear_topk exist? ################"
ls $R/include/ninfer/ops/linear_topk.h 2>&1
grep -rn "linear_topk" $R/src/ops $R/include 2>/dev/null | sed "s|$R/||" | head -10
echo
echo "################ OURS: dspark head branch (dflash_impl.h 455-505) ################"
sed -n '455,505p' $R/src/targets/qwen3_6/impl/runtime/dflash_impl.h
