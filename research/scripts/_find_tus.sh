#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 谁 include dflash2_impl.h ==="
grep -rln "dflash2_impl.h" $R/src $R/apps 2>/dev/null | sed "s|$R/||" | grep -v "\.orig"
echo
echo "=== 谁 include speculative_options.h ==="
grep -rln "speculative_options.h" $R/src $R/apps $R/include 2>/dev/null | sed "s|$R/||" | grep -v "\.orig"
echo
echo "=== 谁 include dflash2_selector.h ==="
grep -rln "dflash2_selector.h" $R/src $R/apps $R/include 2>/dev/null | sed "s|$R/||" | grep -v "\.orig"
echo
echo "=== 谁 include qwen3_6_27b/impl/config.h 或其别名 ==="
grep -rln "targets/qwen3_6_27b/impl/config.h" $R/src $R/apps 2>/dev/null | sed "s|$R/||" | grep -v "\.orig" | head -20
echo
echo "=== 编译入口 (Makefile 目标) ==="
grep -nE "^ninfer|^ninfer-serve|^all:|apps/ninfer" $R/Makefile 2>/dev/null | head -12
