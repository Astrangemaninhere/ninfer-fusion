#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== patchA_build.log 尾部（make ninfer 的结果） ==="
tail -14 "$J/dl/patchA_build.log" | cut -c1-150
echo
echo "=== window_k3 日志 ==="
tail -8 "$J/dl/window_k3.log" | cut -c1-150
echo
echo "=== 二进制时间戳 ==="
ls -l --time-style=+%m-%d_%H:%M /home/user/ninfer-fusion/build/apps/ninfer /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
echo
echo "=== 还在编什么 ==="
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do
  echo "  $(ps -o etime= -p $p | tr -d ' ') $(tr '\0' ' ' < /proc/$p/cmdline | grep -oE '[^ ]+\.(cc|cpp|cu)' | tail -1)"
done
echo
echo "=== 三个 variant TU 是否都重编了（补丁 A 的证据） ==="
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_engine.dir/targets/*/impl/variant.cpp.o 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
