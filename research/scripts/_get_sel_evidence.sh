#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== sel_probe.log 里的 [df2sel] 行（前 20 条）==="
grep -m 20 'df2sel' $J/dl/sel_probe.log 2>/dev/null || echo "(无)"
echo
echo "=== 哪些日志含 df2sel ==="
grep -l 'df2sel' $J/dl/*.log 2>/dev/null || echo "(都不含)"
echo
echo "=== rd_w2.log 那次的完整参数（W=1 得 38%）==="
grep -m3 -oE 'ninfer[^|]{0,200}' $J/dl/rd_w2.log 2>/dev/null | head -3
head -30 $J/dl/rd_w2.log 2>/dev/null | grep -vE '^\s*$' | head -20
