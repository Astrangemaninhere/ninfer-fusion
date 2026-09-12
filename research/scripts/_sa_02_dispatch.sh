#!/bin/bash
set -u
R=/home/user/ninfer-fusion/src/ops
echo "###### wrapper/gdn_input_proj.cpp ######"
cat -n $R/wrapper/gdn_input_proj.cpp
echo
echo "###### sizes ######"
wc -l $R/wrapper/gdn_input_proj.cpp $R/linear_attention/gated_delta_net/*.cuh $R/linear_attention/gated_delta_net/*.h $R/linear_attention/gated_delta_net/*.cpp 2>/dev/null
echo
echo "###### gated_delta_net dir ######"
ls -la $R/linear_attention/gated_delta_net/
