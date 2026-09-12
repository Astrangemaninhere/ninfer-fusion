#!/bin/bash
R=/home/user/ninfer-fusion
echo "######## layouts_impl.h 840-865 ########"
sed -n '840,865p' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
echo
echo "######## 全仓: dflash2 与 proposal_head 同时出现的守卫点 ########"
grep -rn "proposal_head" $R/src/product $R/apps $R/src/serve $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h 2>/dev/null \
  | sed "s|$R/||" | grep -v "\.orig:" | head -20
echo
echo "######## config.h: DFlash2Config 段（确认插入锚点） ########"
sed -n '120,152p' $R/src/targets/qwen3_6_27b/impl/config.h
echo
echo "######## dflash2_impl.h 行尾格式 ########"
file -b $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
sed -n '326,349p' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | cat -v | head -26
