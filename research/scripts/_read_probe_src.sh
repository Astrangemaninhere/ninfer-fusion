#!/bin/bash
# 读探针源码，弄清各列的确切语义（避免误读）
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/program_impl.h
echo '=== T3 探针代码（12420..12495）==='
awk 'NR>=12420 && NR<=12495 {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-165
