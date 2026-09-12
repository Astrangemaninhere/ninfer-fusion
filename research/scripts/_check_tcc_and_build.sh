#!/bin/bash
L=/mnt/c/Users/User/Documents/ziqinzhang/dl/vref3.err
echo '=== tcc 自己的错误输出（CalledProcessError 之前的行）==='
N=$(grep -n 'CalledProcessError' "$L" | head -1 | cut -d: -f1)
[ -n "$N" ] && awk -v s=$((N-45)) -v e=$((N+1)) 'NR>=s && NR<=e {print}' "$L" | grep -vE '^\s*$' | tail -30 | cut -c1-200
echo
echo '=== 搜索 tcc 相关关键字 ==='
grep -nE 'tcc|TinyCC|error:|cannot|not found|unrecognized|unknown option' "$L" | grep -viE 'CalledProcessError|File "' | head -15 | cut -c1-200
echo
echo '=== ninfer 构建进度 ==='
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  编排=$(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)"
echo "  进度=$(grep -oE '\[[ 0-9]+%\]' $J/dl/par_build.log 2>/dev/null | tail -1)"
echo "  二进制=$(stat -c '%y' /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | cut -c1-19)"
echo "  MemAvailable=$(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)"
echo "  Swap=$(awk '/SwapFree/{printf "%.1f GB free", $2/1048576}' /proc/meminfo)"
tail -3 $J/dl/build_spill_ssd.log 2>/dev/null | cut -c1-120
