#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
sed -n '12300,12480p' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | tr -d '\r' > "$D/program_impl_dflash2_ingress.txt"
cat -n "$D/program_impl_dflash2_ingress.txt" | head -200
