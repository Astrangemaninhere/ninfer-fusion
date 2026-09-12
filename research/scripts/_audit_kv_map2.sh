#!/bin/bash
# 三处 spec 后端的 ensure_sequence_kv_mapped 完整实参 + backend_tokens 的语义
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/program_impl.h
echo '=== 11985..11998 (MTP) ==='
awk 'NR>=11985 && NR<=11998 {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-130
echo
echo '=== 12172..12188 (DFlash/dspark) ==='
awk 'NR>=12172 && NR<=12188 {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-130
echo
echo '=== 12407..12422 (DFlash2) ==='
awk 'NR>=12407 && NR<=12422 {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-130
echo
echo '=== 实现体 10320..10360 ==='
awk 'NR>=10320 && NR<=10360 {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-130
