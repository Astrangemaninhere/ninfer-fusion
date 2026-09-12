#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "################ OURS dflash2_impl.h 290-360 ################"
sed -n '290,360p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo
echo "################ UPSTREAM dflash_impl.h 237-335 ################"
sed -n '237,335p' $U/src/targets/qwen3_6/impl/runtime/dflash_impl.h
echo
echo "################ PROBE (NINFER_DF2DBG) definition sites ################"
grep -rn "NINFER_DF2DBG" $R/src $R/include $R/apps 2>/dev/null | sed "s|$R/||" | head -20
