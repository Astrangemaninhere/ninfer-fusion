#!/bin/bash
# 找 v100-skinny / ninfer-4090 相关的目录与产物
echo '=== /home/user 下（WSL 侧）==='
ls -d /home/user/*v100* /home/user/*skinny* /home/user/*4090* 2>/dev/null
ls -d /home/user/ninfer* 2>/dev/null
echo
echo '=== ninfer-fusion 内部的 build/target 目录名 ==='
ls -d /home/user/ninfer-fusion/build* 2>/dev/null
find /home/user/ninfer-fusion -maxdepth 3 -type d \( -name '*v100*' -o -name '*skinny*' -o -name '*4090*' \) 2>/dev/null | head -20
echo
echo '=== 构建脚本里出现的架构关键词 ==='
grep -rlniE 'v100|skinny|4090|sm_70|compute_70|sm_89|compute_89' /home/user/ninfer-fusion/CMakeLists.txt /home/user/ninfer-fusion/src/CMakeLists.txt /home/user/ninfer-fusion/cmake 2>/dev/null | head -10
echo
echo '=== CMake 里的 CUDA 架构设置 ==='
grep -rniE 'CMAKE_CUDA_ARCHITECTURES|CUDA_ARCHITECTURES|sm_70|sm_89|sm_120|compute_70|compute_89' /home/user/ninfer-fusion/CMakeLists.txt /home/user/ninfer-fusion/src/CMakeLists.txt 2>/dev/null | head -15 | cut -c1-140
