#!/bin/bash
R=/home/user/ninfer-fusion
echo "######## 1. materialize: optimized_proposal 填充点 ########"
grep -n -B4 -A10 "optimized_proposal" $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp | sed -n '1,80p'
echo
echo "######## 2. layouts_impl.h: proposal_scratch 的调用点 ########"
grep -n "proposal_scratch" $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
echo
echo "######## 3. layouts_impl.h: dflash2 recipe 段 ########"
grep -n "dflash2" $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h | head -25
echo
echo "######## 4. ops::linear wrapper: Q4G64 是否已注册 ########"
F=$(grep -rl "void linear" $R/src/ops/wrapper/linear.cpp 2>/dev/null | head -1)
echo "file: $F"
grep -n "Q4G64\|Q4G64_F16S" $R/src/ops/wrapper/linear.cpp 2>/dev/null | head -8
echo
echo "######## 5. dflash2_impl.h 里 execution 可见字段（proposal_head?） ########"
grep -rn "proposal_head" $R/src/targets/qwen3_6/impl/runtime/schedule.h $R/src/targets/qwen3_6/impl/runtime/program.h 2>/dev/null | head -8
