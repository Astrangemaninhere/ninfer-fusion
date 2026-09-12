#!/bin/bash
# 回退残缺的 A3_headwise_gate（它引用了全树不存在的 headwise_gate_shape），保留其余 5 条。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"

echo '=== 反向回退 headwise_gate ==='
cp "$C/A3_spark_headwise_gate.diff" /tmp/hw.diff
patch -R -p1 -d "$R" < /tmp/hw.diff 2>&1 | tail -8
echo "  rc=$?"
echo
echo '=== 校验符号已消失 ==='
grep -c 'headwise_gate_shape' "$R/src/ops/wrapper/sigmoid_mul.cpp" 2>/dev/null || echo 0
echo
echo '=== 等当前 make 退出后重编 ==='
while pgrep -x nvcc >/dev/null 2>&1; do sleep 10; done
cd "$R/build" || exit 3
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 30
echo "  nvcc 并行: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
tail -3 "$J/dl/par_build.log" | cut -c1-120
date +%H:%M:%S
