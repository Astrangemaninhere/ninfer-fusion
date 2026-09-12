#!/bin/bash
A=/home/user/ninfer-fusion
B=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "############ config.h diff (B -> A):"
diff -u "$B/src/targets/qwen3_6_27b/impl/config.h" "$A/src/targets/qwen3_6_27b/impl/config.h" | head -80
echo
echo "############ dflash2_impl.h diff (B -> A), first 120 lines:"
diff -u "$B/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" "$A/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" | head -120
