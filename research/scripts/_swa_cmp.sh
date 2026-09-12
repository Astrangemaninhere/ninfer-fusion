#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "################ OURS draft block attention ################"
sed -n '215,292p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo
echo "################ UPSTREAM draft block attention (already seen: 237-275) ################"
sed -n '248,275p' $U/src/targets/qwen3_6/impl/runtime/dflash_impl.h
echo
echo "################ sliding_window_attention signature (public) ################"
sed -n '1,80p' $R/include/ninfer/ops/swa.h
echo
echo "################ sliding_window_attention wrapper impl ################"
F=$(grep -rl "void sliding_window_attention" $R/src/ops/wrapper 2>/dev/null | head -1)
echo "file: $F"
[ -n "$F" ] && cat "$F"
