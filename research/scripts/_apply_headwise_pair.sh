#!/bin/bash
# 配对落地：A3_spark_headwise_gate.diff（功能）+ A3_headwise_gate_fix.diff（补上漏掉的 helper）
# 两片顺序无关、结果字节相同（代理已证 cmp IDENTICAL）。落完 -j8 重编。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
cd "$R" || exit 1

echo '=== 落前状态（确认 headwise 确实不在树）==='
echo "  sigmoid_mul.cpp md5: $(md5sum src/ops/wrapper/sigmoid_mul.cpp | cut -c1-12)  (基线应为 55b41200…)"
echo "  run_headwise_case 出现次数: $(grep -rc 'run_headwise_case' src/ops/wrapper/sigmoid_mul.cpp 2>/dev/null || echo 0)"

apply() {
  local f=$1
  if patch -p1 -b --dry-run < "$f" >/dev/null 2>&1; then patch -p1 -b < "$f" && echo "  applied(raw): $(basename "$f")"; return 0; fi
  local t=/tmp/hw_$(basename "$f"); tr -d '\r' < "$f" > "$t"
  if patch -p1 -b --dry-run < "$t" >/dev/null 2>&1; then patch -p1 -b < "$t" && echo "  applied(cr): $(basename "$f")"; return 0; fi
  echo "  打不上: $(basename "$f")"; return 1
}
echo '=== 配对落库 ==='
apply "$C/A3_spark_headwise_gate.diff" || exit 3
apply "$C/build/A3_headwise_gate_fix.diff" || exit 3
echo "  落完 sigmoid_mul.cpp md5: $(md5sum src/ops/wrapper/sigmoid_mul.cpp | cut -c1-12)"
echo "  headwise_gate_shape 声明+调用: $(grep -c 'headwise_gate_shape' src/ops/wrapper/sigmoid_mul.cpp)"

echo '=== touch 包含者 + -j8 重编 ==='
touch src/ops/wrapper/sigmoid_mul.cpp
cd "$R/build" || exit 4
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 40
echo "  nvcc 并行: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
tail -3 "$J/dl/par_build.log" | cut -c1-120
echo "  par_build 错误: $(grep -cE ' error:' "$J/dl/par_build.log" 2>/dev/null)"
date +%H:%M:%S
