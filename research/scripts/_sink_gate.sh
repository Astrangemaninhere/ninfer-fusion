#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "######## OURS: sink activation sites ########"
grep -rn "capture_positions\|sink.begin\|begin(\|consume_prefill_chunk\|active_tokens\|captured_mask" \
  $R/src/targets/qwen3_6/impl/runtime/ $R/src/targets/qwen3_6/impl/ 2>/dev/null \
  | grep -v "\.ninfer" | sed 's|/home/user/ninfer-fusion/||' | head -40
echo
echo "######## UPSTREAM: sink activation sites ########"
grep -rn "capture_positions\|sink.begin\|begin(\|consume_prefill_chunk\|active_tokens\|captured_mask" \
  $U/src/targets/qwen3_6/impl/runtime/ $U/src/targets/qwen3_6/impl/ 2>/dev/null \
  | sed "s|$U/||" | head -40
echo
echo "######## OURS: capture_layer / capture_positions impl ########"
sed -n '268,332p' $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h
