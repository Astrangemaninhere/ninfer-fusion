#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ 1) speculative_target_impl.h（verify 入口，全文） ################"
cat -n $R/src/targets/qwen3_6/impl/runtime/speculative_target_impl.h
echo
echo "################ 2) TextPhase::Verify 在文本注意力里怎么决定可看的 key ################"
grep -rn "TextPhase::Verify\|phase == .*Verify\|prefill ==" $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h | head -20
echo
echo "################ 3) verify 路径里 valid_columns / rope_positions 的使用 ################"
grep -n "valid_columns\|rope_positions\|cache_positions" $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h | head -30
