#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ DFlash2Config ################"
sed -n '128,175p' $R/src/targets/qwen3_6_27b/impl/config.h
echo
echo "################ sink impl: layers span + feature rows ###############"
sed -n '44,80p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo
echo "################ append_context_impl (context K/V build) ################"
sed -n '95,175p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo
echo "################ frontiers: what positions the context uses ################"
grep -rn "context_frontiers\|execution_frontiers" $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h $R/src/targets/qwen3_6/impl/state/round_state.cpp 2>/dev/null | sed "s|$R/||" | head -20
