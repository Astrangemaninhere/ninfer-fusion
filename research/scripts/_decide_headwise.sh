#!/bin/bash
# 看清 23:33 那次失败的真实报错；若是 headwise 那对补丁（Spark 专属、非关键路径）引起，
# 就地回退它们、保留 head_geometry+gelu_mul，并立刻重编。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
L=$J/dl/par_build.log
export PATH="/home/user/.local/bin:$PATH"

echo '=== 23:33 那次失败的现场（最后 25 行）==='
tail -25 "$L" | cut -c1-140
echo
echo '=== 失败 TU 判定 ==='
if tail -60 "$L" | grep -qE 'sigmoid_mul|sigmoid_gate_mul|headwise'; then
  echo "  => 命中 headwise 相关文件，回退这一对（保留 A3 的另两条）"
  cd "$R" || exit 1
  tr -d '\r' < "$C/build/A3_headwise_gate_fix.diff" > /tmp/f1.diff
  tr -d '\r' < "$C/A3_spark_headwise_gate.diff" > /tmp/f2.diff
  patch -R -p1 < /tmp/f1.diff 2>&1 | tail -2
  patch -R -p1 < /tmp/f2.diff 2>&1 | tail -3
  echo "  回退后 sigmoid_mul.cpp md5: $(md5sum src/ops/wrapper/sigmoid_mul.cpp | cut -c1-12)  (应回到 55b41200…)"
  touch src/ops/wrapper/sigmoid_mul.cpp
  echo '  --- 重编 ---'
  cd "$R/build" || exit 2
  PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
  sleep 35
  echo "  nvcc 并行: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
else
  echo "  => 不是 headwise（见上方现场），不改动；把现场原样留在日志里供下一步判定"
fi
date +%H:%M:%S
