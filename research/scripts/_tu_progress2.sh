#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
L=$J/dl/finish_build.log
date +%H:%M:%S
echo '=== make 日志尾 ==='
tail -8 "$L" | cut -c1-125
echo
echo '=== 正在跑什么（含在编哪个 TU）==='
for p in $(pgrep -x nvcc); do
  echo "  nvcc PID $p 已跑 $(ps -o etimes= -p $p)s  RSS $(( $(ps -o rss= -p $p) / 1024 )) MB"
  tr '\0' '\n' < /proc/$p/cmdline | grep -m1 -E 'CMakeFiles/.*\.(o|cu\.o)$' | sed 's/^/    在编 /'
done
for c in nvlink ptxas cc1plus; do
  for p in $(pgrep -x $c 2>/dev/null); do
    echo "  $c PID $p 已跑 $(ps -o etimes= -p $p)s  RSS $(( $(ps -o rss= -p $p) / 1024 )) MB"
  done
done
echo
echo '=== 库与二进制时间戳 ==='
B=/home/user/ninfer-fusion/build
ls -la --time-style=+%H:%M:%S "$B/lib/libninfer_ops.a" "$B/apps/ninfer" "$B/apps/ninfer-serve" 2>/dev/null | cut -c25-90
echo '  ^ libninfer_ops.a 若比 apps 新 => 库已重建、只差链接'
echo
echo '=== 内存 ==='
free -g | sed -n '2,3p'
