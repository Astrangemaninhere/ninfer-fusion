#!/bin/bash
# A3 headwise_gate 落库后编译报错：sigmoid_mul.cpp:39 'headwise_gate_shape' 未声明。
# 先判：是"缺一个 include"还是"补丁引用了别的补丁才提供的实体"。
R=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo '=== ① 该符号在全树的声明/定义 ==='
grep -rn 'headwise_gate_shape' "$R" 2>/dev/null | head -10 | cut -c1-150
echo
echo '=== ② sigmoid_mul.cpp 的 include 与 30-45 行现场 ==='
awk 'NR<=20 {printf "%4d| %s\n", NR, $0}' "$R/src/ops/wrapper/sigmoid_mul.cpp" | grep -nE 'include' | head -8
echo '  ---'
awk 'NR>=30 && NR<=45 {printf "%4d| %s\n", NR, $0}' "$R/src/ops/wrapper/sigmoid_mul.cpp" | cut -c1-125
echo
echo '=== ③ A3_headwise_gate 补丁到底加了什么（新增文件与 hunk 头）==='
grep -E '^\+\+\+ |^@@' "$C/A3_spark_headwise_gate.diff" | head -20 | cut -c1-120
echo
echo '=== ④ 该补丁新增文件是否在树里 ==='
for f in $(grep -E '^\+\+\+ ' "$C/A3_spark_headwise_gate.diff" | sed 's|^+++ b/||' | sort -u); do
  [ -e "$R/$f" ] && echo "  [在] $f" || echo "  [缺] $f"
done
