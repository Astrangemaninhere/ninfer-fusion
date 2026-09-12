#!/bin/bash
cd /mnt/c/Users/User/Documents/ziqinzhang/_collab/build || exit 1
for f in T1_a5b_revert.diff T1_layout_b_revert.diff T1_dspark_round_probe.diff; do
  add=$(grep -c '^+[^+]' "$f")
  del=$(grep -c '^-[^-]' "$f")
  hunks=$(grep -c '^@@' "$f")
  bytes=$(stat -c %s "$f")
  crlf=$(file -b "$f" | grep -c CRLF)
  echo "$f hunks=$hunks added=$add removed=$del bytes=$bytes crlf=$crlf"
done
echo "--- longest added line width (probe) ---"
awk '/^\+[^+]/{n=length($0); if(n>m){m=n}} END{print "max="m}' T1_dspark_round_probe.diff
echo "--- anchors in probe patch ---"
grep -n 'cstddef\|NINFER_DSPARK_ROUND_LOG\|t1_dspark_round_probe' T1_dspark_round_probe.diff | head
