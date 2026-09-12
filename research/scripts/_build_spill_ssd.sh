#!/bin/bash
# 真正允许溢到 SSD 的构建：
#   ① vm.swappiness 拉满，让内核尽早换出（用户：慢点也行，先把 P 核吃满）
#   ② 构建器内部看门狗关掉（MIN_FREE_GB=0 => 永不触发）
#   ③ 独立守护只在"swap 也快满 + 可用内存极低"这种真 OOM 前夜才杀
# 为什么之前一直挂：看门狗在 MemAvailable<2GB 就杀 nvcc，而 swap 几乎没用 —— 等于自己堵死了 SSD 这条路。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
LOG=$J/dl/build_spill_ssd.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== 允许溢到 SSD 的构建 $(date '+%F %H:%M:%S') ==="
echo "--- 当前内存/交换 ---"
free -g; swapon --show 2>/dev/null | head -3
echo "--- 拉高交换意愿 ---"
sysctl -w vm.swappiness=100 2>&1 | head -2
sysctl -w vm.vfs_cache_pressure=50 2>&1 | head -2

echo "--- 真正的守护：仅当真 OOM 前夜才干预 ---"
guard() {
  while :; do
    av_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    swap_free_kb=$(awk '/SwapFree/{print $2}' /proc/meminfo)
    # 触发条件：可用内存 < 128 MB **且** 剩余 swap < 512 MB
    if [ "$av_kb" -lt 131072 ] && [ "$swap_free_kb" -lt 524288 ]; then
      v=$(ps -eo pid,etimes,comm --no-headers | awk '$3=="nvcc"{print $2, $1}' | sort -n | head -1 | awk '{print $2}')
      if [ -n "${v:-}" ]; then
        echo "  [guard $(date +%H:%M:%S)] 真 OOM 前夜: avail=$((av_kb/1024))MB swap_free=$((swap_free_kb/1024))MB -> 杀 nvcc $v"
        pkill -TERM -P "$v" 2>/dev/null; kill -TERM "$v" 2>/dev/null
      fi
      sleep 20
    elif ! pgrep -x nvcc >/dev/null 2>&1 && ! pgrep -f '_par_build|make' >/dev/null 2>&1; then
      sleep 30
    else
      sleep 10
    fi
  done
}
guard &
GPID=$!
trap 'kill $GPID 2>/dev/null' EXIT

cd "$R/build" || exit 3
echo "--- 并行构建（MIN_FREE_GB=0 => 内部看门狗不介入）$(date +%H:%M:%S) ---"
MIN_FREE_GB=0 PER_JOB_GB=2 RESERVE_GB=0 MAX_JOBS=8 bash "$J/_par_build.sh" ninfer ninfer-serve
echo "  par_build rc=$?"
kill $GPID 2>/dev/null
ls -l --time-style=+%m-%d_%H:%M "$R/build/apps/ninfer" "$R/build/apps/ninfer-serve" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
echo "=== 结束 $(date '+%F %H:%M:%S') ==="
echo BUILD_SPILL_SSD_DONE
