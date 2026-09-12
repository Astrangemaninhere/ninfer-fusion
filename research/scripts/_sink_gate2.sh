#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
for T in "$R|OURS" "$U|UPSTREAM"; do
  ROOT="${T%%|*}"; TAG="${T##*|}"
  echo "######################## $TAG ########################"
  echo "--- DFlashFeatureSink mentions ---"
  grep -rn "DFlashFeatureSink" $ROOT/src $ROOT/include 2>/dev/null \
    | sed "s|$ROOT/||" | grep -v "\.orig:" | head -30
  echo "--- sink API call sites (begin / capture_positions / consume_prefill_chunk) ---"
  grep -rn "\.capture_positions\|capture_positions(\|consume_prefill_chunk\|sink\.begin\|tap\.begin\|feature_sink" \
    $ROOT/src $ROOT/include 2>/dev/null | sed "s|$ROOT/||" | grep -v "\.orig:" | head -30
  echo
done
