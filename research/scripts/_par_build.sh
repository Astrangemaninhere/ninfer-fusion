#!/bin/bash
# 内存感知的并行构建（把 16 核用起来，同时绝不压死机器）。
#
# 为什么需要它：本树历史上一直 -j1（因为并发 nvcc 曾把机器压到只剩 93 MB）。
# 但实测单核利用率极低，而 60 个 TU 里只有 3-4 个是长杆。正确做法不是"永远 -j1"，
# 而是"按可用内存算并发 + 出事时保护机器"。
#
# 三条纪律（都是踩过坑的）：
#   D1 绝不与模型同跑：模型占 ~23 GB 显存 + 大量主机内存，构建立刻 OOM。
#   D2 并发数按 MemAvailable 算，不拍脑袋（见 pick_jobs 表）。
#   D3 看门狗只杀编译器（可重试），不杀模型（用户正在等结果）。
#
# 用法：
#   bash _par_build.sh                      # 默认 make ninfer ninfer-serve
#   bash _par_build.sh ninfer               # 指定目标
#   PER_JOB_GB=6 MAX_JOBS=4 bash _par_build.sh
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/par_build.log
PER_JOB_GB=${PER_JOB_GB:-7}      # 单 TU 经验内存（实测大 TU 尖峰 ~12 GB，但多数可回收）
RESERVE_GB=${RESERVE_GB:-3}      # 留给系统与页面缓存的余量
MAX_JOBS=${MAX_JOBS:-8}
MIN_FREE_GB=${MIN_FREE_GB:-2}    # 低于此值触发保护
TARGETS=${*:-"ninfer ninfer-serve"}
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== par_build $(date '+%F %H:%M:%S')  目标: $TARGETS ==="

mem_avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

# D1: 模型/CLI 在跑就拒绝（否则必 OOM）
if pgrep -f 'apps/ninfer(-serve)? ' >/dev/null 2>&1; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "拒绝启动：检测到模型进程在跑（改用 FORCE=1 可强行启动，风险自负）"
    pgrep -af 'apps/ninfer' | head -3
    exit 3
  fi
  echo "警告：FORCE=1，但仍检测到模型进程"
fi

avail=$(mem_avail_gb)
N=$(( (avail - RESERVE_GB) / PER_JOB_GB ))
[ "$N" -lt 1 ] && N=1
[ "$N" -gt "$MAX_JOBS" ] && N=$MAX_JOBS
echo "MemAvailable=${avail} GB  PER_JOB_GB=${PER_JOB_GB}  预留=${RESERVE_GB} GB  ->  -j${N}"
free -g | sed -n 2p

# D3: 看门狗（后台）——极度紧张时杀掉最年轻的 nvcc 树，保住机器。
WATCHDOG=0
watchdog() {
  while [ "$WATCHDOG" != "1" ]; do
    local av; av=$(mem_avail_gb)
    if [ "$av" -lt "$MIN_FREE_GB" ]; then
      # 最年轻 = 剩余寿命最长 = 损失最小；etimes 升序取第一个
      local v; v=$(ps -eo pid,etimes,comm --no-headers | awk '$3=="nvcc"{print $2, $1}' | sort -n | head -1 | awk '{print $2}')
      if [ -n "$v" ]; then
        echo "  [watchdog $(date +%H:%M:%S)] MemAvailable=${av} GB < ${MIN_FREE_GB} GB -> 杀掉最年轻 nvcc PID $v（make 会重试）"
        pkill -TERM -P "$v" 2>/dev/null
        kill -TERM "$v" 2>/dev/null
        sleep 20
      else
        sleep 5
      fi
    else
      sleep 5
    fi
  done
}
watchdog &
WD_PID=$!
trap 'WATCHDOG=1; kill $WD_PID 2>/dev/null' EXIT

cd "$R/build" || exit 4
rc=1
for target in $TARGETS; do
  t0=$(date +%s)
  echo "--- make $target -j${N}  开始 $(date +%H:%M:%S) ---"
  # 带时间戳记录，便于事后按 TU 算耗时（并行下 mtime 会重叠，必须靠日志）
  make "$target" -j"$N" 2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date +%H:%M:%S)" "$line"
  done
  rc=${PIPESTATUS[0]}
  echo "--- make $target rc=$rc  用时 $(( $(date +%s) - t0 ))s ---"
  [ $rc -ne 0 ] && { echo "（注：若某 TU 被看门狗杀过，重跑一次通常就过）"; }
done
WATCHDOG=1; kill $WD_PID 2>/dev/null
echo "--- 产物 ---"
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" "$R/build/apps/ninfer-serve" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
echo "=== par_build done $(date '+%F %H:%M:%S') rc=$rc ==="
echo PAR_BUILD_DONE
