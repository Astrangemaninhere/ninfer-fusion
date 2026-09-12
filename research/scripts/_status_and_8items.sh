#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== ① ninfer 构建状态 $(date +%H:%M:%S) ==="
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  编排=$(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)"
echo "  进度=$(grep -oE '\[[ 0-9]+%\]' $J/dl/par_build.log 2>/dev/null | tail -1)"
echo "  二进制: $(stat -c '%y  %s bytes' /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | cut -c1-40)"
echo "  ninfer-serve: $(stat -c '%y  %s bytes' /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | cut -c1-40)"
echo "  MemAvailable=$(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)  SwapFree=$(awk '/SwapFree/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
echo "  par_build 尾:"; tail -3 $J/dl/par_build.log 2>/dev/null | cut -c1-120 | sed 's/^/    /'
echo
echo "=== 门后待命链（lane 测试 / 对照）==="
tail -2 $J/dl/lane_tests_gate.log 2>/dev/null | cut -c1-100 | sed 's/^/  /'
echo
echo "=== ② 8 项未落地清单的出处 ==="
ls -la --time-style=+%m-%d_%H:%M $J/_board_append8.md $J/_collab/M_unlanded_now.md 2>/dev/null | cut -c25-95
