#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## _train_df2_shift0.bat（修正配方的训练入口） ##########"
cat "$J/_train_df2_shift0.bat" 2>/dev/null
echo
echo "########## verify_df2head.log ##########"
cat "$J/dl/verify_df2head.log" 2>/dev/null | head -25
echo
echo "########## df2head_stdout.log / df2_copy_stdout.log / df2_copy.log ##########"
tail -12 "$J/dl/df2head_stdout.log" 2>/dev/null
echo "--- copy"
tail -12 "$J/dl/df2_copy.log" 2>/dev/null
echo
echo "########## df2head_plain.log 末尾（接受率汇总） ##########"
grep -E 'acceptance|accepted by pos|decode speed|generated' "$J/dl/df2head_plain.log" 2>/dev/null | tail -8
echo
echo "########## df2head_probe.log 里的接受率/位置剖面 ##########"
grep -E 'acceptance rate|accepted by pos|draft.*per.*round|summary' "$J/dl/df2head_probe.log" 2>/dev/null | tail -10
