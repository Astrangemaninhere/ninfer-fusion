#!/bin/bash
R=/home/user/ninfer-fusion/src/ops/gdn_input_proj
echo "=== 各 GDN conv 变体里的滚动窗口写法（找同样的 s2 = p） ==="
for f in $(ls "$R"/*.cuh "$R"/*.h "$R"/w8/*.cuh "$R"/nvfp4/*.cuh "$R"/q4_q5/*.cuh 2>/dev/null); do
  hits=$(grep -n 's2 = \|s1 = s2\|s0 = s1' "$f" 2>/dev/null | head -6)
  if [ -n "$hits" ]; then
    echo "--- ${f#$R/} ---"
    echo "$hits"
  fi
done
echo
echo "=== 目录内还有哪些 conv 实现文件 ==="
ls "$R" 2>/dev/null
echo "--- w8 / nvfp4 / q4_q5 ---"
ls "$R/w8" "$R/nvfp4" "$R/q4_q5" 2>/dev/null
