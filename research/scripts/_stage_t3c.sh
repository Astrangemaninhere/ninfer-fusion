#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
mkdir -p "$D"
for f in \
  src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h \
  src/ops/kernel/prepare_masked_block.cuh \
  src/ops/launcher/prepare_masked_block.cu \
  src/ops/wrapper/prepare_masked_block.cpp \
  src/ops/kernel/speculative_round.cuh \
  src/ops/launcher/speculative_round.cu \
  src/ops/launcher/speculative_round.h \
  src/ops/wrapper/speculative_round.cpp \
  src/targets/qwen3_6/impl/runtime/speculative_target_impl.h
do
  if [ -f "$R/$f" ]; then
    b=$(basename "$f")
    tr -d '\r' < "$R/$f" > "$D/$b"
    echo "OK  $b $(wc -l < "$D/$b")"
  else
    echo "MISS $f"
  fi
done
echo "=== wrapper dir listing for prepare/spec ==="
ls "$R/src/ops/wrapper/" | grep -iE 'prepare|speculative|masked'
echo "=== launcher headers ==="
ls "$R/src/ops/launcher/" | grep -iE 'prepare|speculative'
echo "=== include/ninfer/ops ==="
ls "$R/include/ninfer/ops/" | grep -iE 'prepare|speculative'
