#!/bin/bash
set -u
R=/home/user/ninfer-fusion/src/ops/gdn_input_proj
for f in nvfp4/nvfp4_gdn_input_plan.h nvfp4/nvfp4_gdn_input_plan.cpp nvfp4/nvfp4_gdn_input_small_t.cu nvfp4/nvfp4_gdn_input_w4a4.cu nvfp4/nvfp4_gdn_input_decode.cu; do
  echo "########## $f ##########"
  cat -n $R/$f
  echo
done
