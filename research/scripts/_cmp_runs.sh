#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
for f in masktok_run.log masktok_run2.log psb_1.log probe_now.log rd_w2.log window_k4.log df2_copy_probe.log; do
  p=$J/dl/$f
  [ -f "$p" ] || { echo "### $f 缺失"; continue; }
  echo "########## $f ##########"
  grep -E 'acceptance|accepted by pos|draft window|rounds|drafted tokens|accepted tokens|decode speed|prompt tokens|generated|tokens' "$p" | head -14
  echo "  [tokens 行] $(grep -m1 'generated ids' "$p" | head -c 120)"
  echo "  行数=$(wc -l < "$p")"
  echo
done
