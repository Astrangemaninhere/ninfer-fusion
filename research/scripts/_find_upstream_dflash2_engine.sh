#!/bin/bash
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
R=/home/user/ninfer-fusion
echo '=== 上游 src 里 dflash2 / DFlash2 的出现（按文件计数）==='
grep -rli 'dflash2' "$U/src" 2>/dev/null | sed "s|$U/||" | sort | head -30
echo
echo '=== 上游 src 里 dflash2 出现总次数 ==='
grep -rli 'dflash2' "$U/src" 2>/dev/null | wc -l
echo
echo '=== 我们 src 里 dflash2 的出现文件数 ==='
grep -rli 'dflash2' "$R/src" 2>/dev/null | grep -v '\.orig' | wc -l
echo
echo '=== 上游是否用 DFlash2Config / kDFlash2 这类符号 ==='
grep -rn 'DFlash2Config\|kDFlash2\|DFlash2Backend\|SpeculativeBackend::DFlash2' "$U/src" 2>/dev/null | head -10 | cut -c1-150
echo
echo '=== 上游 speculative backend 的取值 ==='
grep -rn 'enum class SpeculativeBackend' -A 12 "$U/src" 2>/dev/null | head -18 | cut -c1-130
