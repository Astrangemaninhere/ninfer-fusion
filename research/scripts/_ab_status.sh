#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## A/B 链状态 ##########"
tail -30 $J/dl/df2_ab_chain.log 2>/dev/null
echo
echo "########## 是否还在跑 ##########"
ps -eo etimes,args | grep -E 'apps/ninfer|_df2_ab' | grep -v grep | cut -c1-110
echo
echo "########## 引擎日志里的窗口/精度配置行 ##########"
for L in $J/dl/abz_cur.log $J/dl/df2head_plain.log $J/dl/probe_now.log; do
  [ -f "$L" ] && { echo "--- $L"; grep -iE 'window|hot|kv.dtype|kv dtype|page|keep|precis|fp32' "$L" | head -12; }
done
