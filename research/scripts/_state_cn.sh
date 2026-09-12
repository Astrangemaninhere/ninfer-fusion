#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 编译 ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -3
for p in $(pgrep -f bin/nvcc); do echo "  当前 TU: $(ps -o etime= -p $p | tr -d ' ')"; break; done
grep -c 'patchA build done' "$J/dl/patchA_build.log" 2>/dev/null | sed 's/^/  build 完成标记: /'
echo "=== 测量守护 / K3 ==="
tail -1 "$J/dl/postbuild_measure.log" 2>/dev/null | cut -c1-110
presence=0; pgrep -f '_window_k3.sh' >/dev/null && presence=1; echo "  K3 存活: $presence"
echo "=== Spark 权重下载 ==="
tail -2 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-120
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
echo "=== 内存 ==="
free -g | head -2
