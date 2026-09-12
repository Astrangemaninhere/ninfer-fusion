#!/bin/bash
# 查我们引擎 --kv-dtype / --kv-layer-storage 接受的值（单文件定点 grep）
R=/home/user/ninfer-fusion
echo "=== options.cpp 里 kv-dtype 的解析 ==="
grep -nE '"--kv-dtype"|kv_dtype|"bf16"|"fp8"|"nvfp4"|"e8"|"i8"' $R/apps/cli/options.cpp 2>/dev/null | head -20
echo
echo "=== 该段落的上下文（140-170 行附近） ==="
sed -n '138,166p' $R/apps/cli/options.cpp 2>/dev/null
echo
echo "=== KVDType 枚举定义 ==="
grep -rnE 'enum class .*Kv|KvDType|KVCacheDType|kv_cache_dtype' $R/include/ninfer/types.h $R/src/targets/qwen3_6/impl/runtime/layouts.h 2>/dev/null | head -12
