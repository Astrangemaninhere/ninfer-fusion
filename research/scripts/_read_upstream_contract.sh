#!/bin/bash
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
F=$U/docs/maintainer/qwen3.8-27b-dflash2.md
echo '=== §3 Target hidden conditioning (100-155) ==='
awk 'NR>=100 && NR<=155 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
echo
echo '=== §4 Dynamic grouped convolution (154-186) ==='
awk 'NR>=154 && NR<=186 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
echo
echo '=== §6.3 Path walk 与 proposal distribution (287-320) ==='
awk 'NR>=287 && NR<=320 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
