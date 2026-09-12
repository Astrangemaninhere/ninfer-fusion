#!/bin/bash
set -u
R=/home/user/ninfer-fusion/src/ops
echo "########## linear/nvfp4/nvfp4_config.h ##########"
cat -n $R/linear/nvfp4/nvfp4_config.h
echo
echo "########## grep SmallT consts ##########"
grep -rn "kNvfp4FirstSmallT\|kNvfp4LastSmallT\|kNvfp4TmaRoute\|kInputRows\|kOutputRows" $R/linear/nvfp4/*.h $R/linear/nvfp4/*.cuh $R/linear/nvfp4/*.cpp 2>/dev/null | head -40
echo
echo "########## nvfp4_w4a4_tma_route ##########"
grep -rn -A12 "nvfp4_w4a4_tma_route" $R/linear/nvfp4/*.h $R/linear/nvfp4/*.cuh $R/linear/nvfp4/*.cpp 2>/dev/null | head -60
