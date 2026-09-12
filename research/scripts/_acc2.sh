#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "################ launcher stdout ################"
cat /tmp/da.out 2>/dev/null | tail -20
echo "################ align log ################"
ls -l $J/dl/df2_align*.log $J/dl/df2_probe_raw.log 2>/dev/null
tail -20 $J/dl/df2_align.log 2>/dev/null
echo "################ our dflash2 acceptance call 360-440 ################"
sed -n '360,440p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
