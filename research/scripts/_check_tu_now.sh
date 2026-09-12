#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
date +%H:%M:%S
echo "并行 nvcc 数: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
echo "ptxas 数: $(pgrep -c -x ptxas 2>/dev/null || echo 0)"
echo '--- 进度（百分比）---'
grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -2
echo '--- 编译错误条数 ---'
grep -cE ' error:' "$J/dl/par_build.log" 2>/dev/null
echo '--- 完成/耗时标记 ---'
grep -E 'make .* rc=|PAR_BUILD_DONE|用时|已结束' "$J/dl/par_build.log" 2>/dev/null | tail -4
echo '--- 二进制时间戳 ---'
ls -la --time-style=+%m-%d_%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | cut -c25-80
echo '--- 内存/swap ---'
free -g | sed -n '2,3p'
echo '--- 三路对照链是否已开跑 ---'
tail -3 "$J/dl/post_fix_verify2.log" 2>/dev/null | cut -c1-110
echo '--- 队列表 ---'
ps -ef | grep -E 'pfv2[.]sh|batch_next[.]sh|probe_arm[.]sh|reconfig_par[.]sh|_par_build[.]sh|mem_wd[.]sh' | grep -v grep | awk '{print "  ", $2, $NF}'
