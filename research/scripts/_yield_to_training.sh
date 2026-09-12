#!/bin/bash
# 排他：重训（关键路径）在跑 => 停掉编译，别抢内存。
# 同时确认重训还活着且在推进（它不能被我刚才那轮构建/看门狗误伤）。
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== 停掉我刚起的构建链 ==='
for pid in $(pgrep -f '_par_build[.]sh'); do echo "  kill $pid"; kill $pid 2>/dev/null; done
sleep 2
for p in $(pgrep -x nvcc) $(pgrep -x ptxas); do kill -9 $p 2>/dev/null; done
sleep 2
echo "  剩余 nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"

echo
echo '=== 重训是否还活着并在推进 ==='
tail -3 "$J/dl/train_df2_shift0b.out" 2>/dev/null | cut -c1-120
echo "  python 进程数: $(pgrep -c -x python3 2>/dev/null || echo 0) (WSL 侧)"
echo
echo '=== 内存/显存 ==='
free -g | sed -n '2,3p'
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null
echo
echo '=== 结论 ==='
echo '  编译已让路；重训期间不再启动任何编译或 GPU 跑（避免 0 内存互杀）'
date +%H:%M:%S
