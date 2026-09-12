#!/bin/bash
set -u
R=/home/user/ninfer-fusion
echo "=== gdn_input_proj tree ==="
find $R/src/ops/gdn_input_proj -type f | sort
echo
echo "=== wc -l ==="
find $R/src/ops/gdn_input_proj -type f -name '*' | sort | xargs wc -l 2>/dev/null | tail -40
echo
echo "=== any other gdn dirs ==="
ls -d $R/src/ops/*gdn* $R/src/ops/*Gdn* 2>/dev/null
echo
echo "=== grep gdn includes in src (files only) ==="
grep -rl "gdn" $R/src --include=*.h --include=*.hpp --include=*.cuh --include=*.cu --include=*.cpp 2>/dev/null | sort
