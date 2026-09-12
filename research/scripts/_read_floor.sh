#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/spec_decision.h
echo '=== spec_decision.h 全貌（含地板定义与用法）==='
grep -nE 'floor|kDFlash2|extent|acceptance|0\.05|below|draft' "$F" 2>/dev/null | head -30 | cut -c1-150
echo
echo '=== 相关常量定义 ==='
grep -nE 'constexpr|inline constexpr' "$F" 2>/dev/null | head -20 | cut -c1-140
