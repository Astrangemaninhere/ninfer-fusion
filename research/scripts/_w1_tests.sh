#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== tests mentioning swa or bidirectional:"
grep -rln 'swa\|bidirectional_gqa' "$A/tests" 2>/dev/null | head -20
echo
echo "=== oracle-ish files:"
ls "$A/tests/ops" 2>/dev/null | grep -iE 'swa|bidir|attn' | head -20
echo
for f in $(grep -rln 'bidirectional_gqa_attention(' "$A/tests" 2>/dev/null | head -3); do
  echo "###### $f"
  grep -n 'causal\|Causal\|allow\|oracle\|naive\|reference\|fp64\|double' "$f" | head -40
done
echo
echo "=== does the kernel header mention the test/oracle name?"
grep -rn 'oracle\|Causal' "$A/src/ops/kernel/bidirectional_gqa_attention.cuh" | head
echo
echo "=== swa in tests (any) :"
grep -rn 'swa' "$A/tests" 2>/dev/null | head -20
