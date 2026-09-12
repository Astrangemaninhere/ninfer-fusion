#!/bin/bash
set -u
R=/home/user/ninfer-fusion/src/ops
echo "###### wc wrapper ######"
wc -l $R/wrapper/gdn_input_proj.cpp
echo
echo "###### dispatch keywords in wrapper ######"
grep -n -E "token|Token|T ==|t ==|small_t|decode|verify|chunk|rows|plan|snapshot|\.h\"|auto |if \(|switch" $R/wrapper/gdn_input_proj.cpp | head -150
