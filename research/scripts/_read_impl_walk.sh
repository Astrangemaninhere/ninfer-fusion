#!/bin/bash
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo '=== dflash2_impl.h 300-400（selector / path walk / 取样）==='
awk 'NR>=300 && NR<=400 {printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-155
