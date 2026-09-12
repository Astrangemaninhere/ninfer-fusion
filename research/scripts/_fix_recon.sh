#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "######## A. 上游 linear_topk 签名（可照抄语义） ########"
sed -n '1,60p' $U/include/ninfer/ops/linear_topk.h 2>/dev/null
echo
echo "######## B1. OptimizedProposalWeights / DFlash2Weights 定义 ########"
grep -n -B3 -A16 "struct OptimizedProposalWeights" $R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h
echo "--- DFlash2Weights ---"
grep -n -B2 -A30 "struct DFlash2Weights" $R/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h
echo
echo "######## B2. dspark 的 optimized_proposal 绑定段（照抄模板） ########"
sed -n '440,470p' $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp
echo
echo "######## C. draft_head_rows 定义处 ########"
grep -rn "draft_head_rows" $R/src $R/include 2>/dev/null | sed "s|$R/||" | grep -v orig | head -12
echo
echo "######## D. layouts_impl.h: dflash2 段的 Optimized 处理 ########"
grep -n -B4 -A10 "draft_head_rows" $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h | head -50
echo
echo "######## F. proposal_remap_token_ids 签名 ########"
grep -rn "proposal_remap_token_ids" $R/include $R/src/ops/launcher 2>/dev/null | sed "s|$R/||" | head -6
echo
echo "######## G. 我们的 dflash2 selector 公开签名 ########"
sed -n '1,60p' $R/include/ninfer/ops/dflash2_selector.h
