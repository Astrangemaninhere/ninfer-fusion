#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 谁 include instantiate.h ==="
grep -rln "instantiate.h" $R/src $R/apps 2>/dev/null | sed "s|$R/||" | grep -v "\.orig"
echo
echo "=== build 目录与构建系统 ==="
ls $R/build/CMakeCache.txt $R/build/Makefile 2>&1 | head -3
ls $R/CMakeLists.txt 2>&1
echo
echo "=== 目标名（ninfer 可执行） ==="
grep -rn "add_executable" $R/apps/cli/CMakeLists.txt 2>/dev/null | head -5
grep -rn "name" $R/build/CMakeCache.txt 2>/dev/null | grep -i generator | head -3
echo
echo "=== 现有 .o 时间戳（确认增量基线） ==="
ls -l --time-style=+%m-%d_%H:%M $R/build/apps/ninfer 2>/dev/null
find $R/build -name "*.o" -newer $R/build/apps/ninfer 2>/dev/null | head -5
echo
echo "=== 关掉 ninfer-serve 只编 ninfer 是否可行（看链接依赖） ==="
grep -rn "ninfer-serve\|add_executable(ninfer" $R/apps/*/CMakeLists.txt $R/CMakeLists.txt 2>/dev/null | head -8
