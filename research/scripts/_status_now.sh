#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
B=/home/user/ninfer-fusion/build
date +%H:%M:%S
echo "nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)   ptxas: $(pgrep -c -x ptxas 2>/dev/null || echo 0)"
echo "进度: $(grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1)"
echo "错误条数: $(grep -cE ' error:' "$J/dl/par_build.log" 2>/dev/null)"
echo '--- 最近错误（若有）---'
grep -E ' error:' "$J/dl/par_build.log" 2>/dev/null | tail -3 | cut -c1-150
echo '--- 声明修复是否生效（decoder_state 那条错误还出现吗）---'
grep -c 'kv_rowscale_sidecar_apply_from_env' "$J/dl/par_build.log" 2>/dev/null
echo '--- 二进制 ---'
ls -la --time-style=+%m-%d_%H:%M "$B/apps/ninfer" "$B/apps/ninfer-serve" 2>/dev/null | cut -c25-80
echo '--- 完成标记 / 用时 ---'
grep -E 'make .* rc=|用时|PAR_BUILD_DONE|Built target ninfer$' "$J/dl/par_build.log" 2>/dev/null | tail -4 | cut -c1-110
echo '--- 内存 ---'
free -g | sed -n '2,3p'
echo '--- 三路对照链 ---'
tail -3 "$J/dl/post_fix_verify2.log" 2>/dev/null | cut -c1-110
echo '--- 队列表 ---'
ps -ef | grep -E 'pfv2[.]sh|batch_next[.]sh|probe_arm[.]sh|reconfig_par[.]sh|mem_wd[.]sh|_par_build[.]sh' | grep -v grep | awk '{print "  ", $2, $NF}'
