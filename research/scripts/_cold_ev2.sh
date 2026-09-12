#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== CLI/serve 的 cold-policy 解析 ==="
grep -rnE 'cold.cold?policy|cold_policy' $R/apps/cli/options.cpp $R/src/serve/serve_options.cpp 2>/dev/null | head -20
echo
echo "=== serve_options.cpp 里 host 字面量 ==="
grep -nE '"host"|"window"|"disk"|"none"' $R/src/serve/serve_options.cpp | head -12
echo
echo "=== host_cold_budget 消费点 ==="
grep -rnE 'host_cold_budget|cold_host' $R/include $R/src --include=*.h --include=*.cpp 2>/dev/null | grep -v '\.orig' | head -12
echo
echo "=== effective_cold_pages 全文 ==="
sed -n '115,145p' $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h
