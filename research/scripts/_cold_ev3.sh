#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== program_impl.h:855-885（cold_policy 首个消费点） ==="
sed -n '855,885p' $R/src/targets/qwen3_6/impl/runtime/program_impl.h
echo
echo "=== 既有告警设施 ==="
grep -rnE 'NINFER_WARN|nf_warn|std::cerr|fprintf\(stderr' $R/src/targets/qwen3_6/impl/runtime/*.h 2>/dev/null | grep -v '\.orig' | head -10
echo
echo "=== layouts_impl.h 里 cold_cap 计算点（997 附近） ==="
sed -n '990,1005p' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
