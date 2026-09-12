#!/bin/bash
# 读 packed16 / fp8 / nvfp4 / iso3 / e8 的推导来源，给 fp8-KV 报错定性（只读）
R=/home/user/ninfer-fusion
F=$R/src/ops/wrapper/gqa_attention.cpp
echo "=== 1) 这些标志的定义处（在 160 行之前） ==="
grep -nE 'packed16|const bool fp8|const bool nvfp4|const bool iso3|const bool e8|quant_group|storage' "$F" | head -30
echo
echo "=== 2) 上下文 100-165 ==="
sed -n '100,165p' "$F"
