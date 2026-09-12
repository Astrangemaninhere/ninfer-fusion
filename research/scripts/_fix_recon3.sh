#!/bin/bash
R=/home/user/ninfer-fusion
echo "######## 1. dflash2_proposal_capacity 与 round recipe (layouts_impl.h 686-770) ########"
sed -n '686,772p' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
echo
echo "######## 2. Variant 在本 TU 是否文件级可见 ########"
grep -n "using Variant\|Variant =" $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | head -5
grep -n "Variant::" $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | head -5
echo
echo "######## 3. dspark optimized 分支后的 workspace 是否有 131072 矩阵（对照） ########"
sed -n '570,580p' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
