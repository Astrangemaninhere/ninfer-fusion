#!/bin/bash
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
F=$U/docs/maintainer/qwen3.8-27b-dflash2.md
echo '=== §5 五层 backbone (183-240) ==='
awk 'NR>=183 && NR<=240 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
echo
echo '=== §6.1 Unary candidates + §6.2 Edge score (239-290) ==='
awk 'NR>=239 && NR<=290 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
