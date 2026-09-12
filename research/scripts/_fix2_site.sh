#!/bin/bash
F=/home/user/ninfer-fusion/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
echo "lines=$(wc -l < $F)"
nl -ba "$F" | sed -n '1,75p'
