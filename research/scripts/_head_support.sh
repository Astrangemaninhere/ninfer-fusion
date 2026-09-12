#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
for T in "$R|OURS" "$U|UPSTREAM"; do
  ROOT="${T%%|*}"; TAG="${T##*|}"
  echo "######################## $TAG ########################"
  echo "--- optimized_proposal / ProposalHead / draft_head mentions ---"
  grep -rn "optimized_proposal\|ProposalHead\|draft_head" $ROOT/src $ROOT/include 2>/dev/null \
    | sed "s|$ROOT/||" | grep -v "\.orig:" | head -25
  echo
done
echo "################ OURS: which head does each draft backend use? ################"
grep -rn "output_head" $R/src/targets/qwen3_6/impl/runtime/dflash_impl.h \
    $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
    $R/src/targets/qwen3_6/impl/runtime/mtp_impl.h 2>/dev/null | sed "s|$R/||" | head -20
echo
echo "################ UPSTREAM: which head do the draft backends use (linear_topk callers)? ################"
grep -rn "linear_topk" $U/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | sed "s|$U/||" | head -20
