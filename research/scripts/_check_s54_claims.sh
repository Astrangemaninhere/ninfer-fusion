#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "=== 1) 几何注册表在哪、什么形状（16Q/4KV@D256 是否真的没注册） ==="
find "$R/src" "$R/include" -name 'gqa_attention_geometry*' 2>/dev/null | head -3
sed -n '1,40p' "$R/src/ops/kernel/gqa_attention_geometry.cuh" 2>/dev/null | cut -c1-120
echo
echo "=== 2) 分派处的 KV 头推导（S54 说 16Q 会被读成 2 KV） ==="
sed -n '20,34p' "$R/src/ops/wrapper/gqa_attention.cpp" | cat -n | sed 's/^/  /' | cut -c1-130
sed -n '258,266p' "$R/src/ops/wrapper/gqa_attention.cpp" | cat -n | sed 's/^/  /' | cut -c1-130
echo
echo "=== 3) 生成器的 id 处理（小数点未替换） ==="
grep -n 'replace(' "$R/tools/archkit/adapt.py" | head -6 | cut -c1-140
grep -n 'namespace' "$R/tools/archkit/out/spark-x2.5-4b/config.h" | head -3 | cut -c1-120
echo
echo "=== 4) sigmoid_mul 的同形要求 + 是否存在两输入乘算子 ==="
sed -n '32,44p' "$R/src/ops/wrapper/sigmoid_mul.cpp" 2>/dev/null | cat -n | sed 's/^/  /' | cut -c1-130
ls "$R/include/ninfer/ops/" | grep -iE 'mul|gelu|silu' | head
