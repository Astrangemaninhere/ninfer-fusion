#!/bin/bash
# 逐文件对照：上游 ninfer 的 dflash2 实现 vs 我们树里的自制实现
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
R=/home/user/ninfer-fusion
echo '=== 上游 dflash2 相关源文件 ==='
find "$U/src" -iname '*dflash2*' 2>/dev/null | sed "s|$U/||" | sort | head -30
echo
echo '=== 我们的 dflash2 相关源文件 ==='
find "$R/src" -iname '*dflash2*' 2>/dev/null | sed "s|$R/||" | sort | head -30
echo
echo '=== 上游 runtime 目录里与 dflash2 有关的（含 dflash_context / selector / grouped_conv）==='
find "$U/src" \( -iname '*selector*' -o -iname '*grouped_conv*' -o -iname '*context*' \) 2>/dev/null | sed "s|$U/||" | sort | head -20
echo
echo '=== 上游是否有 dflash2_impl.h（我们那个文件）==='
ls -la "$U/src/targets/qwen3_6/impl/runtime/" 2>/dev/null | head -30 | cut -c30-100
