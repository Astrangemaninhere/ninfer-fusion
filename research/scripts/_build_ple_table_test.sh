#!/bin/bash
# temp: sync PleTable fix + test into the WSL tree, build, run (W2-2 acceptance)
set -e
WIN=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1
SRC=/home/user/ninfer-fusion
SIDE=/mnt/c/Users/User/Documents/ziqinzhang/flashnext_ple

echo "== syncing patched PleTable + test =="
cp "$WIN/src/ops/ple/ple_table.cu" "$SRC/src/ops/ple/ple_table.cu"
cp "$WIN/src/ops/ple/ple_table.h" "$SRC/src/ops/ple/ple_table.h"
cp "$WIN/tools/ple_table_test.cu" "$SRC/tools/ple_table_test.cu"
md5sum "$SRC/src/ops/ple/ple_table.cu" "$SRC/tools/ple_table_test.cu"

echo "== building test =="
CUDA=/usr/local/cuda-13.3/bin/nvcc
cd "$SRC"
"$CUDA" -O2 -std=c++20 -gencode arch=compute_120a,code=[compute_120a,sm_120a] \
  -I src -I third_party -I include -o /home/user/ple_table_test \
  tools/ple_table_test.cu src/ops/ple/ple_layout.cpp src/ops/ple/ple_table.cu 2>&1 | tail -n 15

echo "== running =="
/home/user/ple_table_test --sidecar "$SIDE" --spec "$SIDE/ple_spec.txt" \
  --expect-fnv 0xe8120b70c21aaeb8
