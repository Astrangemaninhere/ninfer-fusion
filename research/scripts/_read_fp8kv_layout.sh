#!/bin/bash
# 读懂 fp8(row256) KV 的布局与 scale 规格，为修 packed16 误判做准备（只读）
R=/home/user/ninfer-fusion
echo "=== 1) layouts_impl.h 里 Fp8E4M3Row256 的分支 ==="
grep -n -A 22 'case KvCacheStorage::Fp8E4M3Row256' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h | head -30
echo
echo "=== 2) gqa_attention.cpp 里 packed16 判定与各家族校验（175-260） ==="
sed -n '160,262p' $R/src/ops/wrapper/gqa_attention.cpp
echo
echo "=== 3) kNvfp4QuantGroup / Row256 相关常量 ==="
grep -rnE 'kNvfp4QuantGroup|Row256|row_scale|kFp8Row' $R/src/ops/wrapper/gqa_attention.cpp $R/include/ninfer/types.h 2>/dev/null | head -12
