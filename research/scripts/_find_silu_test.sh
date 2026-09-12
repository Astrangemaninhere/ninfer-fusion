#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 找 silu 相关测试文件 ==="
find "$R/tests" -iname '*silu*' -o -iname '*swiglu*' 2>/dev/null | head -6
echo
echo "=== test_silu_mul.cpp 结构 ==="
F=$(find "$R/tests" -iname 'test_silu_mul.cpp' 2>/dev/null | head -1)
echo "  file: $F"
[ -n "$F" ] && head -40 "$F" | cat -n | cut -c1-130
echo
echo "=== 用例与断言 ==="
[ -n "$F" ] && grep -nE 'TEST_CASE|SECTION|REQUIRE|CHECK' "$F" | head -14 | cut -c1-130
