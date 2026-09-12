#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== dspark markov argmax 内核（全）==='
F=$R/src/ops/kernel/dspark_markov_argmax.cuh
[ -f "$F" ] && awk '{printf "%4d| %s\n", NR, $0}' "$F" | cut -c1-165
echo
echo '=== 其 launcher/wrapper ==='
grep -rn 'dspark_markov_argmax' $R/src --include=*.cu --include=*.cpp --include=*.h 2>/dev/null | grep -v '\.orig' | head -8 | cut -c1-140
