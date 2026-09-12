#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== ColdPolicy 全量出现点 ==="
grep -rnE 'ColdPolicy' $R/include $R/src --include=*.h --include=*.cpp --include=*.cuh 2>/dev/null | grep -v '\.orig' | head -30
echo
echo "=== Host 枚举定义 ==="
grep -rnE 'enum class ColdPolicy' -A 8 $R/include $R/src --include=*.h 2>/dev/null | grep -v '\.orig' | head -20
