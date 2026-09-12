#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== 在编哪个 TU（确认编译在推进而非卡住） ==="
for p in $(pgrep -f 'bin/nvcc'); do
  echo "  nvcc $(ps -o etime= -p $p | tr -d ' ')  $(tr '\0' ' ' < /proc/$p/cmdline | grep -oE '[^ ]+\.cu' | tail -1)"
done
for p in $(pgrep -f 'cc1plus'); do
  echo "  cc1plus $(ps -o etime= -p $p | tr -d ' ')"
done
echo "  make: $(pgrep -c -f 'make ninfer' 2>/dev/null || echo 0)"
echo "  attempt 日志行数: $(grep -c '' /tmp/reb_ninfer_1.log 2>/dev/null)"
echo "  最近 3 行 make 输出:"
tail -3 /tmp/reb_ninfer_1.log 2>/dev/null | cut -c1-140
echo
echo "=== 对象新鲜度（哪些 TU 已重编） ==="
find $R/build/src/CMakeFiles -name '*.o' -newermt '19:25' -printf '%TH:%TM %8s %p\n' 2>/dev/null | sort | tail -8 | sed 's#/home/user/ninfer-fusion/build/src/CMakeFiles/##' | cut -c1-120
