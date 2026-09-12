#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== $(date +%H:%M:%S) 构建 ==="
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  编排=$(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)"
echo "  进度=$(grep -oE '\[[ 0-9]+%\]' $J/dl/par_build.log 2>/dev/null | tail -1)  新错误=$(grep -cE ' error:' $J/dl/par_build.log 2>/dev/null)"
echo "  二进制=$(stat -c '%y' $R/build/apps/ninfer 2>/dev/null | cut -c1-19)"
echo "  内存=$(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)  交换用=$(awk '/SwapTotal|SwapFree/{printf "%s ", $0}' /proc/meminfo | head -c 80)"
echo "  构建日志尾:"
tail -3 $J/dl/build_spill_ssd.log 2>/dev/null | cut -c1-120 | sed 's/^/    /'
echo "  par_build 尾:"
tail -3 $J/dl/par_build.log 2>/dev/null | cut -c1-120 | sed 's/^/    /'
echo
echo "=== lane 测试门 ==="
tail -3 $J/dl/lane_tests_gate.log 2>/dev/null | cut -c1-110 | sed 's/^/  /'
echo
echo "=== 层号差 1 的线索：草稿 config vs 训练 env ==="
grep -n 'target_layer_ids' "$J/data/draft_model/config.json" 2>/dev/null | cut -c1-90
grep -rn 'hs_layers_env' "$J"/collect-hs* "$J"/*.py 2>/dev/null | head -3 | cut -c1-120
grep -rn 'target_layer_ids\|hs_layers' /mnt/c/Users/User/Documents/ziqinzhang/train_dspark.py 2>/dev/null | head -5 | cut -c1-130
