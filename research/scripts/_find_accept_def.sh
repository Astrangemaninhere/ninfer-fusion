#!/bin/bash
R=/home/user/ninfer-fusion
cd "$R" || exit 1
echo '=== acceptance rate 出现的所有点 ==='
grep -rnE 'acceptance rate|acceptance_rate|speculative_accepted|spec_drafted' apps/ src/ --include=*.cpp --include=*.h 2>/dev/null | grep -v '\.orig' | head -14 | cut -c1-150
echo
echo '=== 逐点取上下文（分子/分母）==='
for f in $(grep -rlE 'acceptance rate' apps/ src/ --include=*.cpp 2>/dev/null | grep -v '\.orig' | head -3); do
  echo "--- $f ---"
  n=$(grep -nE 'acceptance rate' "$f" | head -1 | cut -d: -f1)
  awk -v s=$((n-16)) -v e=$((n+3)) 'NR>=s && NR<=e {printf "%5d| %s\n", NR, $0}' "$f" | cut -c1-135
done
