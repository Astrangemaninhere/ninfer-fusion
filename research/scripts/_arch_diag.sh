#!/bin/bash
# 关键：到底在给几个架构编 ptxas？多架构 = 成倍 ptxas 时间，且 -t/--threads 只能并行架构。
export PATH="/home/user/.local/bin:$PATH"
B=/home/user/ninfer-fusion/build
echo '=== 正在运行的 nvcc 完整命令行 ==='
ps -eo args= -C nvcc | head -1 | tr ' ' '\n' | grep -nE 'generate-code|arch|code=|maxrregcount|ptxas|O[0-9]|threads|split|lineinfo|dopt' | head -25
echo
echo '=== CMakeCache 里的架构与编译选项 ==='
grep -iE '^CMAKE_CUDA_ARCHITECTURES|^CMAKE_CUDA_FLAGS|^CMAKE_BUILD_TYPE|^CMAKE_CUDA_COMPILER_LAUNCHER' "$B/CMakeCache.txt" | cut -c1-160
echo
echo '=== 当前 GPU 与算力 ==='
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv 2>/dev/null
echo
echo '=== 编译命令来源：flags.make 里的 nvcc 选项 ==='
for d in "$B"/src/CMakeFiles/ninfer_ops.dir "$B"/src/CMakeFiles/ninfer_nvfp4_tma.dir; do
  [ -f "$d/flags.make" ] || continue
  echo "--- $d/flags.make ---"
  grep -E 'CUDA_FLAGS|CUDA_DEFINES' "$d/flags.make" | cut -c1-400
done
echo
echo '=== 有几种 sm 目标会被真正编出来（看已生成的 .o 大小与 cubin）==='
grep -rhoE '\-gencode[= ][^ ]*' "$B"/src/CMakeFiles/ninfer_ops.dir/flags.make 2>/dev/null | sort -u | head
grep -c . "$B"/src/CMakeFiles/ninfer_ops.dir/flags.make 2>/dev/null
