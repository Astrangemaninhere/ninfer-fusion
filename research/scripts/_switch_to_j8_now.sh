#!/bin/bash
# 立刻把 -j1 的构建切成 -j8 激进并行（用户要求，且实测还剩 109 条待编译/链接）。
# 纪律：
#   * 杀之前先打印 cmdline（铁律②）；
#   * 杀掉在飞的 .o 与 .o.d —— 否则截断的对象比源文件"新"，make 会当它已更新而跳过，
#     把半成品链接进去（这比慢一点危险得多）；
#   * 驱动脚本也要杀，否则它会看到 make 非零返回而用 -j1 重试（attempt 2）。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
B=$R/build
export PATH="/home/user/.local/bin:$PATH"

echo "=== 杀前取证 ==="
for p in $(pgrep -x nvcc); do
  echo "  nvcc PID $p 已跑 $(ps -o etimes= -p $p)s"
  tr '\0' '\n' < /proc/$p/cmdline | grep -E '^-o$' -A1 | tail -1 | sed 's/^/    -o /'
  tr '\0' '\n' < /proc/$p/cmdline | grep -m1 -E '^/home.*\.cu$' | sed 's/^/    输入 /'
done
INFLIGHT=$(for p in $(pgrep -x nvcc); do tr '\0' '\n' < /proc/$p/cmdline | grep -m1 -E 'CMakeFiles/.*\.o$'; done | head -1)
echo "  在飞对象: ${INFLIGHT:-（未识别）}"

echo
echo "=== 杀构建链（驱动 -> make -> nvcc/ptxas）==="
for pid in $(pgrep -f '_rebuild_after_s3[5]'); do echo "  kill 驱动 $pid"; kill $pid 2>/dev/null; done
pkill -f 'ninfer-fusion/build' 2>/dev/null
sleep 2
for p in $(pgrep -x nvcc) $(pgrep -x ptxas); do kill -9 $p 2>/dev/null; done
sleep 2
echo "  剩余 nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)  ptxas: $(pgrep -c -x ptxas 2>/dev/null || echo 0)"

echo
echo "=== 删掉在飞半成品 ==="
if [ -n "$INFLIGHT" ] && [ -f "$B/$INFLIGHT" ]; then
  ls -la --time-style=+%H:%M:%S "$B/$INFLIGHT"
  rm -f "$B/$INFLIGHT" "$B/$INFLIGHT.d"
  echo "  已删（make 会重编它）"
else
  echo "  未识别到在飞对象，跳过（保守：不动任何 .o）"
fi

echo
echo "=== 以 -j8 激进并行重开（允许溢到 SSD swap）==="
free -g | sed -n 2p
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 MIN_FREE_GB=0.5 \
  setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 25
echo "--- 25 秒后：并行度是否上来了 ---"
echo "  nvcc 数: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
ps -eo etimes,rss,comm | grep -E 'nvcc|cc1plus' | head -8
free -g | sed -n '2,3p'
echo "--- 并行构建日志头 ---"
tail -6 "$J/dl/par_build.log" 2>/dev/null | cut -c1-120
date +%H:%M:%S
