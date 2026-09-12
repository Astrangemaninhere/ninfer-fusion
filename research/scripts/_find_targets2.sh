#!/bin/bash
# 有界搜索（每个都有 timeout + 限定目录深度），detached 跑，结果写日志
J=/mnt/c/Users/User/Documents/ziqinzhang
{
echo "=== $(date +%H:%M:%S) 有界搜索 ==="
echo '--- ① 顶层文档里出现 skinny/4090 的 ---'
timeout 20 grep -rl -i -e skinny -e 4090 "$J"/*.md 2>/dev/null | head -10
echo '--- ② _collab 文档里 ---'
timeout 20 grep -rl -i -e skinny -e 4090 "$J/_collab"/*.md 2>/dev/null | head -10
echo '--- ③ WSL /home/user 顶层目录名 ---'
timeout 10 ls -d /home/user/*skinny* /home/user/*v100* /home/user/*4090* 2>/dev/null
echo '--- ④ ninfer-fusion 树内目录名（限深度 3，带超时）---'
timeout 30 find /home/user/ninfer-fusion -maxdepth 3 -type d \( -iname '*skinny*' -o -iname '*v100*' -o -iname '*4090*' \) 2>/dev/null | head -15
echo '--- ⑤ 源码里的架构关键词（限 src/，带超时）---'
timeout 40 grep -rniE 'skinny|v100|4090' /home/user/ninfer-fusion/src/CMakeLists.txt /home/user/ninfer-fusion/CMakeLists.txt 2>/dev/null | head -10 | cut -c1-140
echo '--- ⑥ CMake 架构设置 ---'
timeout 20 grep -rniE 'CMAKE_CUDA_ARCHITECTURES|sm_70|sm_89|sm_90|sm_120' /home/user/ninfer-fusion/CMakeLists.txt 2>/dev/null | head -8 | cut -c1-140
echo "=== done $(date +%H:%M:%S) ==="
} > "$J/dl/find_targets.log" 2>&1
echo FIND_TARGETS_DONE >> "$J/dl/find_targets.log"
