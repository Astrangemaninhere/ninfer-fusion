#!/bin/bash
# 独立内存看门狗：_par_build.sh 里那句 [: 0.5: integer expected 让内置看门狗没起来，
# 这里补一个不受影响的（用 MB 整数比较，避免小数）。
# 策略：MemAvailable < 250 MB 时杀掉最年轻的 nvcc（make 会重编它），保住机器；
#       绝不杀模型（用户可能在等结果）——只对 nvcc/ptxas 下手。
set -u
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/mem_watchdog.log
exec >> "$LOG" 2>&1
echo "=== standalone watchdog 起 $(date '+%F %H:%M:%S') ==="
low_hits=0
while :; do
  # 构建都结束了就退出（没有 nvcc/make 时留 3 次确认）
  if ! pgrep -x nvcc >/dev/null 2>&1 && ! pgrep -f '_par_build|make' >/dev/null 2>&1; then
    low_hits=$((low_hits + 1))
    [ $low_hits -ge 3 ] && { echo "=== 构建已结束，看门狗退出 $(date +%H:%M:%S) ==="; exit 0; }
    sleep 20; continue
  fi
  low_hits=0
  av_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$av_kb" -lt 262144 ]; then   # < 256 MB
    v=$(ps -eo pid,etimes,comm --no-headers | awk '$3=="nvcc"{print $2, $1}' | sort -n | head -1 | awk '{print $2}')
    if [ -n "${v:-}" ]; then
      echo "[$(date +%H:%M:%S)] MemAvailable=$((av_kb/1024)) MB -> 杀最年轻 nvcc PID $v"
      pkill -TERM -P "$v" 2>/dev/null; kill -TERM "$v" 2>/dev/null
      sleep 20
    else
      sleep 10
    fi
  else
    sleep 15
  fi
done
