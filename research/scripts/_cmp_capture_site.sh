#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '######## 我们 text_context_impl.h:1215-1258 ########'
awk 'NR>=1215 && NR<=1258 {printf "%4d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | cut -c1-150
echo
echo '######## 上游 text_context_impl.h:1022-1062 ########'
awk 'NR>=1022 && NR<=1062 {printf "%4d| %s\n", NR, $0}' "$U/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | cut -c1-150
