#!/bin/bash
A=/home/user/ninfer-fusion
B=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
for f in src/targets/qwen3_6/impl/runtime/program_impl.h \
         src/targets/qwen3_6/impl/runtime/layouts_impl.h \
         src/product/speculative_options.h \
         src/targets/qwen3_6_27b/impl/variant.cpp ; do
  echo "############ $f : diffstat"
  diff -u "$B/$f" "$A/$f" | grep -cE '^[+-]' 
  echo "--- hunks touching draft/dflash2/proposal/attention_valid/steps:"
  diff -u "$B/$f" "$A/$f" | grep -nE '^[+-].*(dflash2|draft|proposal|attention_valid|steps|block_drafts|extent)' | head -40
done
