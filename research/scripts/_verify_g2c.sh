#!/bin/bash
G=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 我自己复核：整套检查器（5 个模块） ==="
cd "$G" && python3 gui_i18n_check.py 2>&1 | tail -8
echo
echo "=== 复核 {mb} 那个 bug 是否真修了（不该再有裸 {mb}） ==="
grep -n '{mb}' "$G/model_import.py" | head -5 | cut -c1-140
grep -n 'nosize\|head_tied_true' "$G/model_import.py" | head -4 | cut -c1-140
echo
echo "=== 最近被改的 GUI 文件 ==="
find "$G" -maxdepth 1 -type f -newermt '-25 minutes' -printf '%TH:%TM %8s %f\n' | sort
echo
echo "=== 编译 / 测量守护 ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -2
for p in $(pgrep -f bin/nvcc); do echo "  当前 TU 已编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
tail -1 "$J/dl/postbuild_measure.log" | cut -c1-100
echo
echo "=== Spark 权重 ==="
tail -2 "$J/dl/spark_weights.log" | tr -d '\r' | cut -c1-120
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
