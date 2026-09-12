#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 源码 vs 二进制的 mtime ==="
stat -c '%y  %n' $R/src/targets/qwen3_6_27b/impl/config.h $R/build/apps/ninfer 2>&1
echo
echo "=== 二进制里可见的符号/字符串线索（config 是 constexpr，肉眼看不出数值）==="
echo "--- config.h 两个 mask_token 各自属于哪个类 ---"
awk 'NR>=85 && NR<=160 { if ($0 ~ /struct |class |^};/) print NR": "$0; else if ($0 ~ /mask_token/) print NR": "$0 }' $R/src/targets/qwen3_6_27b/impl/config.h
echo
echo "=== dflash2_impl.h 用的是哪个 Config ==="
grep -n 'Config::mask_token\|namespace Config\|using Config\|Config =\|#include .*config' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | head -20
echo
echo "=== 目标文件里 mask_token 是否已被编入（常量传播后只剩 248070/248077 立即数）==="
find $R/build -name 'config*.o' -newer $R/src/targets/qwen3_6_27b/impl/config.h 2>/dev/null | head
echo "(上面为空 ⇒ config.h 比目标文件新 ⇒ 需要重编)"
